import os
import glob
import time
import datetime
import psutil
import subprocess
import random
from threading import Thread
import logging

import GPUtil
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)
app = Flask(__name__)
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class STATUS:
    WAITING = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3
    CLOSED = 4


class Task:

    def __init__(self,
                 job_id,
                 path,
                 command,
                 num_gpus=0,
                 delay=0,
                 min_gpu_memory=None,
                 output_path: str = None) -> None:
        self.num_gpus = num_gpus
        self.path = path
        self.command = command
        self.job_id = job_id
        self.min_gpu_memory = min_gpu_memory
        self.delay = delay
        self.file_stream = None
        self.proc = None
        self.status = STATUS.WAITING
        self.uptime = "N/A"
        self.used_gpus = set()

        if output_path is None:
            os.makedirs("outputs", exist_ok=True)
            self.output_path = f"outputs/{self.job_id}.log"

    def poll(self):
        if self.proc is not None:
            return self.proc.poll()
        else:
            return None

    def run(self):
        self.status = STATUS.RUNNING
        self.file_stream = open(self.output_path, "w")
        cwd = os.getcwd()
        os.chdir(self.path)
        self.proc = subprocess.Popen(self.command, stdout=self.file_stream, stderr=self.file_stream, shell=True)
        os.chdir(cwd)

    def close(self):
        if self.proc is None:
            logger.warning("Task is not started yet!")
        elif self.proc is not None and self.proc.poll() is None:
            raise ValueError("Process not done yet!")
        else:
            self.proc = None
            self.used_gpus = set()

    def check_status(self):
        res = self.proc.poll() if self.proc is not None else None
        if res is None:
            return self.status
        elif res == 0:
            self.status = STATUS.DONE
            logger.info(f"Task {self.job_id} is done!")
        elif res > 0:
            self.status = STATUS.ERROR
            logger.error(f"Task {self.job_id} has error!")
        return self.status

class Scheduler(Thread):

    def __init__(self) -> None:
        super().__init__()
        self.max_num_gpus = 8
        output_files = glob.glob(os.path.join(os.path.dirname(__file__), "outputs/*.log"))
        if len(output_files) > 0:
            self._job_count = max([int(file.split("/")[-1].split(".")[0]) for file in output_files])
        else:
            self._job_count = 0
        
        self.used_gpus = set()
        self.tasks = []

    def run(self):
        while True:
            # iterate through all gpus
            # GPUs = GPUtil.getGPUs()
            deviceIDs = GPUtil.getAvailable(order='random',
                                            limit=8,
                                            maxLoad=0.2,
                                            maxMemory=0.5,
                                            includeNan=False,
                                            excludeID=[],
                                            excludeUUID=[])
            deviceIDs = list(set(deviceIDs) - self.used_gpus)

            if len(self.tasks) < 1:
                logger.info("No task is in the pool!")

            for task in self.tasks:
                if task.status == STATUS.WAITING:
                    logger.info("Waiting to allocate resource for task!")
                    break

            all_tasks_completed_flag = True
            for task in self.tasks:
                if task.status == STATUS.DONE or task.status == STATUS.ERROR:
                    all_tasks_completed_flag = all_tasks_completed_flag & True
                else:
                    all_tasks_completed_flag = all_tasks_completed_flag & False
            if all_tasks_completed_flag and len(self.tasks) > 0:
                logger.info("All task completed!")

            # loop over satisfied tasks
            for task in self.tasks:
                if task.status == STATUS.WAITING and task.num_gpus <= len(deviceIDs):
                    # execute the task if can be run
                    if task.num_gpus > 0:
                        logger.info(f"GPU {deviceIDs} are available! Selecting the first one.")
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
                        self.used_gpus.add(deviceIDs[0])
                        task.used_gpus.add(deviceIDs[0])

                    logger.info("Running task!")
                    
                    
                    task.run()
                    # give the command at least 30 seconds before running the next experiment
                    time.sleep(task.delay)
                    break

            # check all tasks statues
            for task in self.tasks:
                if task.status != STATUS.DONE and task.status != STATUS.ERROR:
                    status = task.check_status()
                    if status == STATUS.DONE or status == STATUS.ERROR:
                        self.used_gpus -= task.used_gpus
                        task.close()

                # if len(tasks) > 0 and task_flag == False:
                #     logger.info("No")
                # else:

            time.sleep(2)

    def submit(self, path, command, delay, *args, **kwargs):
        self._job_count += 1
        task = Task(job_id=self._job_count, path=path, command=command, delay=delay, num_gpus=kwargs["num_gpus"])
        self.tasks.append(task)
        return f"Job {self._job_count} is submitted!"

    def get_tasks_stats(self):
        stats = []
        for task in self.tasks:
            # get status
            if task.status == STATUS.WAITING:
                status = "Waiting"
            elif task.status == STATUS.RUNNING:
                status = "Running"
            elif task.status == STATUS.DONE:
                status = "Done"
            elif task.status == STATUS.ERROR:
                status = "Error"

            # get uptime
            if task.proc != None:
                try:
                    pid = task.proc.pid
                    p = psutil.Process(pid)
                    start_time = p.create_time()
                    diff_time = time.time() - start_time
                    task.uptime = str(datetime.timedelta(seconds=int(diff_time)))
                except:
                    task.proc = None

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


@app.route('/')
def index():
    return render_template('server_table.html', title='Workload Manager')

@app.route('/view_log/<job_id>')
def read_log(job_id):
    with open(f"outputs/{job_id}.log") as f:
        data = f.read()
    return render_template('log.html', title='Workload Manager', content=data)


@app.route('/api/update_table')
def update_table():
    processes = scheduler.get_tasks_stats()
    total_num_processes = len(processes)

    # search filter
    search = request.args.get('search[value]')
    filtered_processes = []
    if len(search) > 0:
        for p in processes:
            if search in str(p["ID"]) or search in p["command"]:
                filtered_processes.append(p)
        processes = filtered_processes
    total_filtered = len(processes)

    # sorting
    col_index = request.args.get(f'order[0][column]')
    if col_index is not None:
        col_name = request.args.get(f'columns[{col_index}][data]')

        if col_name not in ['ID', 'command', 'status', "path"]:
            col_name = 'ID'

        descending = request.args.get(f'order[0][dir]') == 'desc'

        processes = sorted(processes, key=lambda x: x[col_name], reverse=descending)

    # pagination
    start = request.args.get('start', type=int)
    length = request.args.get('length', type=int)
    processes = processes[start:start + length]

    # response
    return {
        'data': [p for p in processes],
        'recordsFiltered': total_filtered,
        'recordsTotal': total_num_processes,
        'draw': request.args.get('draw', type=int),
    }


@app.route('/update_process', methods=['GET', 'POST'])
def update_process():
    path = request.form['path']
    command = request.form['command']
    num_gpus = int(request.form['gpus'])
    delay = float(request.form['delay'])
    message = scheduler.submit(path=path, command=command, delay=delay, num_gpus=num_gpus)
    return message


if __name__ == '__main__':
    scheduler = Scheduler()
    scheduler.daemon = True
    scheduler.start()
    time.sleep(0.5)
    app.run(host="0.0.0.0", port=18812)