import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

signalPath = "../../misc/data/exp03_B_signal/"
outputPath = "./"

fileNameListAll = []

for x in ["a1"]:
    for a in range(5):
        for b in range(1, 5):
            fileNameListAll.append(x + "_" + str(a) + "_" + str(b))

# fileNameList = ["1_1_1"]
fileNameList = fileNameListAll

nPreSignal = 100
nSignal = 500
nSample = 1016

if __name__ == "__main__":
    try:
        if os.path.isdir(outputPath) is False:
            os.mkdir(outputPath)
            print(
                "Successfuly created an unexist folder! outputPath= " +
                outputPath)
            print("")

        meanList = []
        stdList = []

        for idx in tqdm(range(len(fileNameList)), desc="PROCESSING", ncols=100, unit=" file"):
            fileName = fileNameList[idx]
            distance = np.load(signalPath + fileName + "_distance.npy")

            meanList.append(np.mean(distance))
            stdList.append(np.std(distance))

            # hist = np.zeros(len(signalI[0]))
            #
            # for idx in tqdm(range(nSignal), desc=fileName, ncols=100, unit=" signal"):
            #     for x in range(len(signalI[idx])):
            #         if signalI[idx][x] > outlier or signalQ[idx][x] > outlier:
            #             hist[x] += 1
            #
            with plt.style.context(['science', 'ieee', 'grid']):
                plt.rcParams.update({"font.family": "serif", "font.serif": ["Times"], "font.size": 6})
                plt.plot(distance, color="#5975A4")
                # plt.bar(np.arange(len(signalI[0])), hist, color="#5975A4")

                plt.title(fileName.replace("_", "-"))
                plt.savefig(outputPath + fileName + ".png")
                plt.close()

        print(meanList)
        print(stdList)

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print("[outlier:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
