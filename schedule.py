import os
import GPUtil


deviceIDs = GPUtil.getAvailable(order = 'random', limit = 5, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])

os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
os.system("echo $CUDA_VISIBLE_DEVICES")