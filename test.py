import os
import sys
import subprocess
import logging
import time
import termios
import atexit

import GPUtil


logger = logging.getLogger(__name__)

class STATUS:
    WAITING = 0
    RUNNING = 1
    CLOSED = 2

class Task:
    def __init__(self, output_path: str = None) -> None:
        self.working_dir = None
        self.command = None
        self.job_id = None
        self.file_stream = None
        self.proc = None
        self.min_gpu_memory = None
        self.status = STATUS.WAITING

        if output_path is None:
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
        os.chdir(self.working_dir)
        self.proc = subprocess.Popen(self.command, stdout=sys.stdout, shell=True)
        os.chdir(cwd)

    def close(self):
        self.status = STATUS.CLOSED
        if self.proc.poll() is None:
            logger.warn("Process not done yet!")
        else:
            self.file_stream.close()


class Scheduler:

    def __init__(self) -> None:
        self.tasks = []

    def run(self):
        while True:
            # iterate through all gpus
            # GPUs = GPUtil.getGPUs()
            deviceIDs = GPUtil.getAvailable(order = 'first', limit = 5, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])

            # assign a task
            if GPUavailability[0] == 1:
                pass
            time.sleep(0.5)

    def add_task(self, task):
        self.tasks.append(task)


deviceIDs = GPUtil.getAvailable(order = 'first', limit = 5, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])

print("xxx")



class Terminal:
    def __init__(self) -> None:
        # initialize the terminal
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.listener_mode = True
        atexit.register(self.on_exit)


    def check_interrupt(self, trainer):
        key = os.read(sys.stdin.fileno(), 1)

        if key != b'' and key != None:

            if key == b'~':
                self.listener_mode = False
                self.call_console(trainer)
                self.listener_mode = True

            # clear stdin buffer
            while key != b'' and key != None:
                key = os.read(sys.stdin.fileno(), 1)

    @property
    def listener_mode(self):
        return self._listener_mode

    @listener_mode.setter
    def listener_mode(self, value):
        if not isinstance(value, bool):
            raise NotImplementedError("Not boolean type!")

        if not hasattr(self, "_listener_mode"):
            self._listener_mode = False

        if value == True and self._listener_mode == False:
            self.old_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)  # lflags
            new_settings[6][termios.VMIN] = 0  # cc
            new_settings[6][termios.VTIME] = 0  # cc
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
        elif value == False and self._listener_mode == True:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

        self._listener_mode = value

    def call_console(self):
        # TODO: Build a more interactive system
        logger.warning(
            """\n
                          #####################################\n
                          ##### Entering Scheduler Console! #####\n
                          #####################################\n"""
        )

        logger.warning("PDB is implemented here!")
        breakpoint()

    def on_exit(self):
        # it is important to restore the terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)



# result = os.system("python task1.py")

# print("running command")

# f = open()

# # (out, err) = proc.communicate()
# # print(out)

# # f = open("blah.txt", "w")
# # subprocess.call(["/home/myuser/run.sh", "/tmp/ad_xml",  "/tmp/video_xml"], stdout=f)

# print("test")

# while proc.poll() is None:
#     # print("not done")
#     pass

# print(proc.poll())
