import os

for i in range(20):
    os.system("../scheduler/flyrun -g 1 -d 10 --min_gpu_memory 5000 python task1.py")