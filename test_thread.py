import os
import time
from threading import Thread

class Scheduler(Thread):

    def __init__(self) -> None:
        super().__init__()
        self.tasks = []
        self.a = 1

    def run(self):
        while True:
            print("a", self.a)
            time.sleep(1)

scheduler = Scheduler()
scheduler.start()

time.sleep(5)
scheduler.a = 9999


scheduler.join()