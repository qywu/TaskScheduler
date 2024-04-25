import subprocess

file_stream = open("output.txt", "w")
proc = subprocess.Popen("python test.py", shell=True, stdout=file_stream, stderr=file_stream, bufsize=1)

print("Doing something else...")
while True:
    pass

file_stream.close()
