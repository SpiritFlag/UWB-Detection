from src.global_vars import *
from src.preprocess.global_vars import *

import os
import sys
import numpy as np
from tqdm import tqdm

class SignalExtraction():
    def __init__(self):
        super(SignalExtraction, self).__init__()

    def main(self):
        try:
            assert os.path.isdir(orgSignalPath),\
                f"orgSignalPath= {orgSignalPath} "\
                "does not exist.\n"

            if os.path.isdir(extractSignalPath) is False:
                os.mkdir(extractSignalPath)
                print(
                    "Successfuly created an unexist folder! extractSignalPath= " +
                    extractSignalPath)
                print("")

            for x in tqdm(range(len(fileNameList)), desc="PROCESSING", ncols=100, unit=" file"):
                fileName = fileNameList[x]

                for rep in range(1, repetition + 1):
                    filepath = orgSignalPath + fileName + str(rep) + ".dat"

                    _, _, fast_length = np.fromfile(filepath, dtype=np.int32, count=3)
                    raw_arr = np.fromfile(filepath, dtype=np.float32, count=-1)

                    slow_length = len(raw_arr) // (fast_length + 3)

                    raw_arr = raw_arr.reshape((slow_length, fast_length + 3))
                    # meta_arr = np.frombuffer(raw_arr[:, :3].tobytes(), dtype=np.int32).reshape((slow_length, 3))
                    data_arr = raw_arr[:, 3:]

                    np.save(extractSignalPath + fileName + str(rep) + "_signal",
                        np.array(data_arr))

            print("")

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[SignalExtraction:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")


# class SignalExtraction():
#     def __init__(self):
#         super(SignalExtraction, self).__init__()
#
#     def main(self):
#         try:
#             assert os.path.isdir(orgSignalPath),\
#                 f"orgSignalPath= {orgSignalPath} "\
#                 "does not exist.\n"
#
#             if os.path.isdir(extractSignalPath) is False:
#                 os.mkdir(extractSignalPath)
#                 print(
#                     "Successfuly created an unexist folder! extractSignalPath= " +
#                     extractSignalPath)
#                 print("")
#
#             for fileName in fileNameList:
#                 file = open(orgSignalPath + fileName + ".log", "r")
#
#                 for x in range(nPreSignal):
#                     while True:
#                         line = file.readline()
#                         if len(line) > 0 and line[:10] == "Accum Len ":
#                             break
#
#                 signalListI = []
#                 signalListQ = []
#                 distanceList = []
#
#                 for x in tqdm(range(nSignal), desc=fileName, ncols=100,
#                     unit=" signal"):
#                     signalI = []
#                     signalQ = []
#
#                     while True:
#                         line = file.readline()
#                         if len(line) > 0 and line[:10] == "Anchor ToF":
#                             distanceList.append(float(line[27:35]))
#
#                             while True:
#                                 line = file.readline()
#                                 if len(line) > 0 and line[:10] == "Accum Len ":
#                                     for y in range(nSample):
#                                         line = [float(i) for i in \
#                                             file.readline().rstrip("\n").split(", ")]
#                                         signalI.append(line[0])
#                                         signalQ.append(line[1])
#                                     break
#
#                             signalListI.append(signalI)
#                             signalListQ.append(signalQ)
#
#                             break
#
#                 np.save(extractSignalPath + fileName + "_Isignal",
#                     np.array(signalListI))
#                 np.save(extractSignalPath + fileName + "_Qsignal",
#                     np.array(signalListQ))
#                 np.save(extractSignalPath + fileName + "_distance",
#                     np.array(distanceList))
#
#             print()
#
#         except Exception as ex:
#             _, _, tb = sys.exc_info()
#             print("[SignalExtraction:" + str(tb.tb_lineno) + "] " + str(ex) +
#                 "\n\n")
