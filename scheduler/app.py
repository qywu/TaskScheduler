import os
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

        if output_path is None:
            os.makedirs("outputs", exist_ok=True)
            self.output_path = f"outputs/{self.job_id}.txt"

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
        self.proc = subprocess.Popen(self.command, stdout=self.file_stream, shell=True)
        os.chdir(cwd)

    def close(self):
        if self.proc.poll() is None:
            raise ValueError("Process not done yet!")


class Scheduler(Thread):

    def __init__(self) -> None:
        super().__init__()
        self.max_num_gpus = 8
        self._job_count = 0
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
                    logger.info(f"GPU {deviceIDs} are available! Selecting the first one.")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
                    logger.info("Running task!")
                    task.run()
                    # give the command at least 30 seconds before running the next experiment
                    time.sleep(task.delay)
                    break

            # check all tasks statues
            for task in self.tasks:
                res = task.poll()
                if res is None:
                    continue
                elif res == 0:
                    task.status = STATUS.DONE
                    logger.info(f"Task {task.job_id} is done!")
                    task.close()
                elif res > 0:
                    task.status = STATUS.ERROR
                    logger.error(f"Task {task.job_id} has error!")
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

            item = {
                "ID": task.job_id,
                "uptime": task.uptime,
                "path": task.path,
                "command": task.command,
                "status": status,
            }
            stats.append(item)
        return stats


@app.route('/')
def index():
    return render_template('server_table.html', title='Workload Manager')


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
    app.run(port=18812)