import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

signalPath = "../../misc/data/exp01_B_signal/"
outputPath = "fig/"

fileNameListAll = []

for x in range(1, 5):
    fileNameListAll.append(str(x))

# fileNameList = ["1"]
fileNameList = fileNameListAll

nSignal = 500

if __name__ == "__main__":
    try:
        if os.path.isdir(outputPath) is False:
            os.mkdir(outputPath)
            print(
                "Successfuly created an unexist folder! outputPath= " +
                outputPath)
            print("")

        for fileName in fileNameList:
            signalI = np.load(signalPath + fileName + "_Isignal.npy")
            signalQ = np.load(signalPath + fileName + "_Qsignal.npy")

            for idx in tqdm(range(nSignal), desc=fileName, ncols=100, unit=" signal"):
                ax = plt.axes(projection="3d")
                ax.plot3D(signalI[idx], signalQ[idx], np.arange(len(signalI[idx])), color="#5975A4")

                name = fileName + "--" + str(idx)
                plt.title(name)
                plt.savefig(outputPath + name + ".png")
                plt.close()

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print("[fig_3D:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
