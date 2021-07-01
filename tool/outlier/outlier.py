import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

signalPath = "../../misc/data/exp01_B_signal/"
outputPath = "./"

fileNameListAll = []

for x in range(1, 5):
    fileNameListAll.append(str(x))

# fileNameList = ["1"]
fileNameList = fileNameListAll

nSignal = 500
outlier = 1000

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

            hist = np.zeros(len(signalI[0]))

            for idx in tqdm(range(nSignal), desc=fileName, ncols=100, unit=" signal"):
                for x in range(len(signalI[idx])):
                    if signalI[idx][x] > outlier or signalQ[idx][x] > outlier:
                        hist[x] += 1

            with plt.style.context(['science', 'ieee', 'grid']):
                plt.rcParams.update({"font.family": "serif", "font.serif": ["Times"], "font.size": 6})
                plt.bar(np.arange(len(signalI[0])), hist, color="#5975A4")

                plt.title(fileName)
                plt.savefig(outputPath + fileName + ".png")
                plt.close()

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print("[outlier:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
