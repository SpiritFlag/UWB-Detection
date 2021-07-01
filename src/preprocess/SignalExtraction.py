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

            for fileName in fileNameList:
                file = open(orgSignalPath + fileName + ".txt", "r")

                for x in range(nPreSignal):
                    while True:
                        line = file.readline()
                        if len(line) > 0 and line[:10] == "Accum Len ":
                            break

                signalListI = []
                signalListQ = []

                for x in tqdm(range(nSignal), desc=fileName, ncols=100,
                    unit=" signal"):
                    signalI = []
                    signalQ = []

                    while True:
                        line = file.readline()
                        if len(line) > 0 and line[:10] == "Accum Len ":
                            for y in range(nSample):
                                line = [float(i) for i in \
                                    file.readline().rstrip("\n").split(", ")]
                                signalI.append(line[0])
                                signalQ.append(line[1])
                            break

                    signalListI.append(signalI)
                    signalListQ.append(signalQ)

                np.save(extractSignalPath + fileName + "_Isignal",
                    np.array(signalListI))
                np.save(extractSignalPath + fileName + "_Qsignal",
                    np.array(signalListQ))

            print()

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[SignalExtraction:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")
