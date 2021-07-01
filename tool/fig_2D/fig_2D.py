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

            with plt.style.context(['science', 'ieee', 'grid']):
                plt.rcParams.update({"font.family": "serif", "font.serif": ["Times"], "font.size": 6})

                for idx in tqdm(range(nSignal), desc=fileName, ncols=100, unit=" signal"):
                    plt.plot(signalI[idx], signalQ[idx], ".", markersize=0.5, color="#5975A4")

                    name = fileName + "--" + str(idx)
                    plt.title(name)
                    plt.savefig(outputPath + name + ".png")
                    plt.close()

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print("[fig_2D:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
