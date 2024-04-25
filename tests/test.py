import os
import time

count = 0
while True:
    count += 1
    print(f"Hello, World! {count}", flush=True)
    time.sleep(0.1)