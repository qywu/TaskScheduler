import os
import time

count = 0
for i in range(100):
    count += 1
    print(f"Hello, World! {count}", flush=True)
    time.sleep(0.1)