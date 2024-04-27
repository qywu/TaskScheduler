import os
import argparse
import socket
import signal
import glob
import time
import datetime
from omegaconf import OmegaConf
import psutil
import subprocess
import random
import threading
from threading import Thread, Lock
import queue
import logging

import GPUtil

import asyncio
import aiofiles

from flask import (
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    Response,
    stream_with_context,
)
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required


class User(UserMixin):
    def __init__(self, id):
        self.id = id


config = OmegaConf.load("config.yaml")

# Assuming there's only one user
users = config["user"]

logger = logging.getLogger(__name__)
app = Flask(__name__, static_url_path="/static")
app.secret_key = b"test_secret_key"

login_manager = LoginManager(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None


logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def get_hostname_safe() -> str:
    try:
        hostname = socket.gethostname()
        return hostname
    except Exception as e:
        print(f"An error occurred while fetching the hostname: {e}")
        return ""


HOSTNAME = get_hostname_safe()


class TASK_STATUS:
    WAITING = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3
    KILLED = 4


def terminate_process_tree(pid, timeout=5):
    """
    Attempts to terminate a process and all its children processes. If they do not exit within
    the specified timeout, forcefully kills them.

    Args:
        pid (int): Process ID of the parent process to terminate.
        timeout (int): Time in seconds to wait for processes to exit gracefully.

    Returns:
        None
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)  # Get all child processes

        # Try to terminate all child processes first
        for child in children:
            child.terminate()

        # Wait for child processes to terminate
        gone, still_alive = psutil.wait_procs(children, timeout=timeout)
        for p in still_alive:
            logger.warning(f"Child process {p.pid} did not terminate, killing it.")
            p.kill()

        # After handling children, try to terminate the parent process
        parent.terminate()
        parent.wait(timeout=timeout)  # Wait for the parent process to terminate

        # If the parent process hasn't terminated, kill it
        if parent.is_running():
            logger.warning(f"Parent process {pid} did not terminate, killing it.")
            parent.kill()
            logger.info(f"Parent process {pid} killed.")
        else:
            logger.info(f"Parent process {pid} terminated gracefully.")

    except psutil.NoSuchProcess:
        logger.error(f"Process {pid} not found!")
    except psutil.AccessDenied:
        logger.error(f"Access denied when trying to terminate process {pid}.")
    except Exception as e:
        logger.error(
            f"An error occurred while trying to terminate process {pid}: {str(e)}"
        )


class Task:
    def __init__(
        self,
        job_id,
        path,
        command,
        num_gpus=0,
        gpus=None,
        time_interval=0,
        min_gpu_memory=None,
        max_num_retries=0,
        log_path: str = None,
    ) -> None:
        self.job_id = job_id
        self.path = path
        self.command = command
        self.num_gpus = num_gpus
        self.gpus = gpus
        self.min_gpu_memory = min_gpu_memory
        self.time_interval = time_interval
        self.pid = None
        self.proc = None
        self.status = TASK_STATUS.WAITING
        self.uptime = "N/A"
        self.used_gpus = set()
        self.start_time = None
        self.position_in_queue = -1
        self.max_num_retries = max_num_retries
        self.remaining_retries = max_num_retries

        if log_path is None:
            os.makedirs("outputs", exist_ok=True)
            self.log_path = os.path.join("outputs", f"{job_id}.log")
        else:
            self.log_path = log_path

    def run(self):
        try:
            self.file_stream = open(self.log_path, "a")
            self.status = TASK_STATUS.RUNNING
            self.proc = subprocess.Popen(
                self.command,
                stdout=self.file_stream,
                stderr=self.file_stream,
                shell=True,
                cwd=self.path,
                bufsize=1,
            )
            self.pid = self.proc.pid
            self.start_time = psutil.Process(self.pid).create_time()
        except Exception as e:
            self.status = TASK_STATUS.ERROR
            logger.error(f"Failed to start task {self.job_id}: {e}")

    def poll(self):
        if self.pid:
            return self.proc.poll()
        return None

    def close(self):
        if self.pid:
            terminate_process_tree(self.pid)

        self.pid = None
        self.used_gpus.clear()
        if hasattr(self, "file_stream"):
            self.file_stream.close()

    def check_and_update_status(self):
        if self.pid:
            return_code = self.poll()
            if return_code is not None:
                if return_code == 0:
                    self.status = TASK_STATUS.DONE
                    logger.info(f"Task {self.job_id} completed successfully!")
                else:
                    self.status = TASK_STATUS.ERROR
                    logger.error(
                        f"Task {self.job_id} failed with return code {return_code}"
                    )
                return self.status
        return self.status

    def reset(self):
        self.pid = None
        self.proc = None
        self.status = TASK_STATUS.WAITING
        self.uptime = "N/A"
        self.used_gpus = set()
        self.start_time = None
        self.position_in_queue = -1


class Scheduler(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.daemon = True
        self.lock = Lock()

        self.used_gpus = set()
        self.tasks = []
        # Enable all GPUs by default
        self.gpus = GPUtil.getGPUs()
        self.enabled_gpus = {int(gpu.id): True for gpu in self.gpus}
        self.max_num_gpus = len(self.gpus)
        self.last_run_time = time.time()
        self.wait_time_interval = -1

    def run(self):
        while True:
            self.gpus = GPUtil.getGPUs()
            self.check_and_run_tasks()
            self.check_task_statuses()

            time.sleep(config.scheduler.check_interval)

    def check_and_run_tasks(self):
        if not self.tasks:
            return

        # check if the last task was run time_interval seconds ago
        if time.time() - self.last_run_time < self.wait_time_interval:
            return

        avail_gpus = [
            (gpu, gpu.memoryFree) for gpu in self.gpus if self.enabled_gpus[int(gpu.id)]
        ]
        avail_gpus.sort(
            key=lambda x: x[1], reverse=True
        )  # Sort GPUs by available memory, use the most free GPU first

        # Loop over all tasks
        for task in self.tasks:
            # update uptime
            if task.status == TASK_STATUS.RUNNING:
                start_time = task.start_time
                diff_time = time.time() - start_time
                task.uptime = str(datetime.timedelta(seconds=int(diff_time)))
            
            if task.status == TASK_STATUS.WAITING:
                logger.info(f"Task {task.job_id} is waiting for resource allocation!")
                # Filter GPUs based on task-specific GPU ids and sufficient memory
                suitable_gpus = [
                    gpu
                    for gpu, mem in avail_gpus
                    if int(gpu.id) in task.gpus  # Ensure GPU is in task's allowed list
                    and mem >= task.min_gpu_memory
                    # and int(gpu.id) not in self.used_gpus
                ]

                if len(suitable_gpus) >= task.num_gpus:
                    self.run_task(task, suitable_gpus)
                    self.last_run_time = time.time()
                    self.wait_time_interval = task.time_interval
                    break

    def run_task(self, task, suitable_gpus):
        gpu_ids = [int(gpu.id) for gpu in suitable_gpus[: task.num_gpus]]
        logger.info(f"Allocating GPUs {gpu_ids} to task {task.job_id}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        self.used_gpus.update(gpu_id for gpu_id in gpu_ids)
        task.used_gpus.update(gpu_id for gpu_id in gpu_ids)

        task.run()
        # Give the command to run time_interval before running the next task
        # e.g. to avoid running multiple tasks at the same time and to give the task some time to start up
        # time.sleep(task.time_interval)

    def check_task_statuses(self):
        all_tasks_completed = all(
            task.status in {TASK_STATUS.DONE, TASK_STATUS.ERROR, TASK_STATUS.KILLED}
            for task in self.tasks
        )

        if all_tasks_completed and self.tasks:
            logger.info("All tasks completed! Please check the logs for results.")

        for task in self.tasks:
            if task.status not in {
                TASK_STATUS.DONE,
                TASK_STATUS.ERROR,
                TASK_STATUS.KILLED,
            }:
                status = task.check_and_update_status()
                if status in {TASK_STATUS.DONE, TASK_STATUS.ERROR, TASK_STATUS.KILLED}:
                    self.release_resources(task)

                if status == TASK_STATUS.ERROR:
                    if task.remaining_retries > 0:
                        logger.warning(
                            f"Task {task.job_id} failed! Retrying {task.remaining_retries} more times."
                        )
                        task.reset()
                        task.job_id = f"{task.job_id}_retry_{task.max_num_retries - task.remaining_retries}"
                        self.tasks.append(task)
                        task.remaining_retries -= 1
                    else:
                        logger.error(
                            f"Task {task.job_id} failed! No more retries left."
                        )

    def sort_tasks_by_status(self):
        # Define a dictionary to map statuses to sorting priorities
        priority = {
            TASK_STATUS.DONE: 1,
            TASK_STATUS.ERROR: 2,
            TASK_STATUS.RUNNING: 3,
            TASK_STATUS.WAITING: 3,
            TASK_STATUS.KILLED: 5,
        }

        # Sort the tasks list using the priority of their statuses
        self.tasks.sort(key=lambda task: priority[task.status])

    def release_resources(self, task):
        self.used_gpus.difference_update(task.used_gpus)
        task.used_gpus.clear()
        task.close()

    def update_waiting_task_positions(self):
        waiting_count = 0  # Counter for tasks in waiting status
        for task in self.tasks:
            if task.status == TASK_STATUS.WAITING:
                waiting_count += 1
                task.position_in_queue = waiting_count

    def submit(self, path, command, time_interval, min_gpu_memory, *args, **kwargs):
        timestamp = datetime.datetime.now().isoformat()
        job_id = f"{HOSTNAME}_{timestamp}"
        gpus = kwargs["gpus"]
        if not gpus or len(gpus) == 0:
            gpus = list(range(self.max_num_gpus))

        task = Task(
            job_id=job_id,
            path=path,
            command=command,
            time_interval=time_interval,
            num_gpus=kwargs["num_gpus"],
            min_gpu_memory=min_gpu_memory,
            gpus=gpus,
        )
        self.tasks.append(task)
        return f"Job {task.job_id} is submitted!"

    def get_tasks_stats(self):
        self.update_waiting_task_positions()
        stats = []
        for task in self.tasks:
            # get status
            if task.status == TASK_STATUS.WAITING:
                status = f"Waiting {task.position_in_queue} tasks"
            elif task.status == TASK_STATUS.RUNNING:
                status = "Running"
            elif task.status == TASK_STATUS.DONE:
                status = "Done"
            elif task.status == TASK_STATUS.ERROR:
                status = "Error"
            elif task.status == TASK_STATUS.KILLED:
                status = "Killed"
            else:
                status = "Unknown"

            used_gpus = str(list(task.used_gpus))[1:-1]

            item = {
                "ID": task.job_id,
                "uptime": task.uptime,
                "gpus": used_gpus,
                "path": task.path,
                "command": task.command,
                "status": status,
            }
            stats.append(item)
        return stats

    def cleanup(self):
        "Remove finished tasks"
        new_tasks = []
        for task in self.tasks:
            if task.status not in {
                TASK_STATUS.DONE,
                TASK_STATUS.ERROR,
                TASK_STATUS.KILLED,
            }:
                new_tasks.append(task)
        self.tasks = new_tasks


from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired()])
    password = PasswordField("Password", validators=[InputRequired()])


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and users[username]["password"] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid credentials", "danger")
    return render_template("login.html", form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
def index():
    return render_template("server_table.html", title="Task Scheduler")


async def stream_text_file(filename, idle_timeout=30):
    async with aiofiles.open(filename, mode="r") as file:
        await file.seek(0)  # Move to the beginning of the file
        start_idle_time = None  # Track start of idle time due to empty lines

        while True:
            try:
                # Wait for readline with a timeout
                line = await asyncio.wait_for(
                    file.readline(), config.server.stream_timeout
                )

                if not line:
                    if start_idle_time is None:
                        start_idle_time = (
                            asyncio.get_running_loop().time()
                        )  # Mark start of idle time
                    elif (
                        asyncio.get_running_loop().time() - start_idle_time
                    ) > idle_timeout:
                        logger.info(f"Idle timeout reached while reading {filename}")
                        break  # Break if the idle timeout is exceeded
                    await asyncio.sleep(0.1)  # Sleep briefly if line is empty
                    continue

                # Reset idle timer on reading data
                start_idle_time = None
                yield line.encode("utf-8")  # Encode line for streaming
            except asyncio.TimeoutError:
                # Break the loop if readline times out
                logger.info(f"Timeout reading {filename}")
                break


def stream_file(filename, stop_event):
    q = queue.Queue()

    def run_async_loop():
        asyncio.run(push_to_queue(filename, q, stop_event))

    async def push_to_queue(filename, q, stop_event):
        try:
            async for line in stream_text_file(filename):
                if stop_event.is_set():
                    break
                q.put(line)
            q.put(None)  # Signal the end of streaming
        except Exception as e:
            print("An error occurred:", e)
            q.put(None)

    thread = Thread(target=run_async_loop, daemon=True)
    thread.start()

    while not stop_event.is_set():
        line = q.get()
        if line is None:
            break
        yield line

    thread.join()
    logger.info(f"Stopped streaming {filename}")


@app.route("/stream_log/<job_id>")
def stream(job_id):
    stop_event = threading.Event()
    filename = os.path.join(os.path.dirname(__file__), f"outputs/{job_id}.log")
    response = Response(
        stream_with_context(stream_file(filename, stop_event)),
        mimetype="text/event-stream",
    )
    response.call_on_close(
        lambda: stop_event.set()
    )  # Set stop event when client disconnects
    return response


@app.route("/view_log/<job_id>")
def read_log(job_id):
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), f"outputs/{job_id}.log")
    ):
        return "Log file not found!"
    return render_template("log.html", title="Task Scheduler", job_id=job_id)


# def try_terminate_then_kill(pid, timeout=10):
#     """
#     Tries to terminate the process with the given PID and kills it if it doesn't exit within the timeout period.

#     Args:
#         pid (int): Process ID of the task to terminate.
#         timeout (int): Time in seconds to wait for the process to exit gracefully before killing it.

#     Returns:
#         None
#     """
#     try:
#         proc = psutil.Process(pid)
#         # Try to terminate the process
#         proc.terminate()
#         # Wait for the process to terminate
#         try:
#             proc.wait(timeout=timeout)
#             logger.info(f"Process {pid} terminated gracefully.")
#         except psutil.TimeoutExpired:
#             # If the process is still alive after the timeout, kill it
#             proc.kill()
#             logger.warning(f"Process {pid} was killed after timeout.")
#     except psutil.NoSuchProcess:
#         logger.error(f"Process {pid} not found!")
#     except Exception as e:
#         logger.error(f"An error occurred while trying to terminate process {pid}: {str(e)}")


@app.route("/kill_job/<job_id>")
def kill_job(job_id):
    flag = False
    for task_idx, task in enumerate(scheduler.tasks):
        if task.job_id == job_id:
            flag = True
            break

    if flag:
        if task.status == TASK_STATUS.RUNNING:
            if task.pid is not None:
                task.close()
                logger.info(f"Killed {job_id} PID: {task.pid}")
                task.status = TASK_STATUS.KILLED
        else:
            logger.info(
                f"Task {job_id} is not running, cannot kill it. removing it from the task list."
            )
            scheduler.tasks.remove(task)

    # flash(f"Killed job {job_id}!", "danger")
    return redirect("/")


@app.route("/gpus_info")
def get_gpus_info():
    return render_template("gpus_info.html", title="Task Scheduler")


@app.route("/update_enabled_gpus", methods=["GET", "POST"])
def update_enabled_gpus():
    if request.method == "POST":
        gpu_id = int(request.form["gpu_id"])
        enabled = request.form["enabled"] == "false"
        scheduler.enabled_gpus[gpu_id] = not enabled
        flash(
            f"Changed GPU {gpu_id} status to {scheduler.enabled_gpus[gpu_id]}!",
            "warning",
        )
        logger.info(f"Changed GPU {gpu_id} status to {scheduler.enabled_gpus[gpu_id]}!")
        return redirect("/gpus_info")
    else:
        return redirect("/gpus_info")


@app.route("/cleanup", methods=["GET", "POST"])
def cleanup():
    scheduler.cleanup()
    flash("Successfully cleaned up!", "success")
    return redirect(url_for("index"))


@app.route("/api/update_gpus_table")
def update_gpus_table():
    data = {"results": []}
    GPUs = GPUtil.getGPUs()

    for gpu_id, gpu in enumerate(GPUs):
        item = {
            "gpu_id": gpu.id,
            "memory": f"{str(int(gpu.memoryUsed))+'MB'}/"
            + str(int(gpu.memoryTotal))
            + "MB",
            "utilize": f"{int(gpu.load*100)}%",
            "enabled": scheduler.enabled_gpus[int(gpu.id)],
        }
        data["results"].append(item)

    return data


@app.route("/api/update_processes_table")
def update_processes_table():
    processes = scheduler.get_tasks_stats()
    total_num_processes = len(processes)

    # search filter
    search = request.args.get("search[value]")
    filtered_processes = []
    if len(search) > 0:
        for p in processes:
            if search in str(p["ID"]) or search in p["command"]:
                filtered_processes.append(p)
        processes = filtered_processes
    total_filtered = len(processes)

    # sorting
    col_index = request.args.get(f"order[0][column]")
    if col_index is not None:
        col_name = request.args.get(f"columns[{col_index}][data]")

        if col_name not in ["ID", "command", "status", "path"]:
            col_name = "ID"

        descending = request.args.get(f"order[0][dir]") == "desc"

        processes = sorted(processes, key=lambda x: x[col_name], reverse=descending)

    # pagination
    start = request.args.get("start", type=int)
    length = request.args.get("length", type=int)
    processes = processes[start : start + length]

    # response
    return {
        "data": [p for p in processes],
        "recordsFiltered": total_filtered,
        "recordsTotal": total_num_processes,
        "draw": request.args.get("draw", type=int),
    }


@app.route("/update_process", methods=["GET", "POST"])
def update_process():
    path = request.form["path"]
    command = request.form["command"]
    num_gpus = int(request.form["num_gpus"])
    gpus = [
        int(gpu_id) for gpu_id in request.form["gpus"].split(",") if gpu_id.isdigit()
    ]
    time_interval = float(request.form["time_interval"])
    min_gpu_memory = int(request.form["min_gpu_memory"])
    max_num_retries = int(request.form["max_num_retries"])
    password = request.form["password"]
    if password != config.user["admin"]["password"]:
        return "Invalid password!"
    message = scheduler.submit(
        path=path,
        command=command,
        time_interval=time_interval,
        num_gpus=num_gpus,
        gpus=gpus,
        min_gpu_memory=min_gpu_memory,
        max_num_retries=max_num_retries,
    )
    return message


if __name__ == "__main__":
    scheduler = Scheduler()
    scheduler.start()

    print(
        f"Starting server at {config.server.host}:{config.server.port}\n"
        "    please visit http://{config.server.host}:{config.server.port} for logs."
    )

    import textwrap

    cmd_example = """
        To submit a task, you can use the following command to request 1 gpu with 30GB gpu memory and allow 10 seconds of time interval:
            taskrun -n 1 -m 30000 -t 10 python test.py
        """
    cmd_example = textwrap.dedent(cmd_example)

    print(cmd_example)

    app.run(host=config.server.host, port=config.server.port, debug=False)
