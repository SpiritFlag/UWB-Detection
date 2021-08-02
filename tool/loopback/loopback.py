import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

signalPath = "../../misc/data/exp03_B_signal/"
outputPath = "fig/"

fileNameListAll = []
classification = 4
repetition = 4

for a in ["a1"]:
    for b in range(1, classification + 1):
        fileNameListAll.append(a + "_" + str(b) + "_")

fileNameList = ["a1_1_1"]
# fileNameList = fileNameListAll

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

        for fileName in fileNameList:
            signalI = np.load(signalPath + fileName + "_Isignal.npy")
            signalQ = np.load(signalPath + fileName + "_Qsignal.npy")

            fig = plt.figure(figsize=[34, 13])

            ax1 = fig.add_subplot(221)
            ax1 = sns.heatmap(pd.DataFrame(signalI), cmap="viridis")
            plt.title("Before Filtering - I")
            plt.xlabel("Sample Index")
            plt.ylabel("Frame Index")
            plt.xticks(np.arange(0, nSample, 100), np.arange(0, nSample, 100))
            plt.yticks(np.arange(0, nSignal, 100), np.arange(0, nSignal, 100))

            ax2 = fig.add_subplot(222)
            ax2 = sns.heatmap(pd.DataFrame(signalQ), cmap="viridis")
            plt.title("Before Filtering - Q")
            plt.xlabel("Sample Index")
            plt.ylabel("Frame Index")
            plt.xticks(np.arange(0, nSample, 100), np.arange(0, nSample, 100))
            plt.yticks(np.arange(0, nSignal, 100), np.arange(0, nSignal, 100))

            filteredI = []
            filteredQ = []

            filteredI.append(signalI[0])
            filteredQ.append(signalQ[0])

            alpha = 0.99

            for idx in tqdm(range(1, nSignal), desc=fileName, ncols=100, unit=" signal"):
                filteredI.append(alpha * filteredI[-1] + (1 - alpha) * signalI[idx])
                filteredQ.append(alpha * filteredQ[-1] + (1 - alpha) * signalQ[idx])

            ax3 = fig.add_subplot(223)
            ax3 = sns.heatmap(pd.DataFrame(filteredI), cmap="viridis")
            plt.title("After Filtering - I")
            plt.xlabel("Sample Index")
            plt.ylabel("Frame Index")
            plt.xticks(np.arange(0, nSample, 100), np.arange(0, nSample, 100))
            plt.yticks(np.arange(0, nSignal, 100), np.arange(0, nSignal, 100))

            ax4 = fig.add_subplot(224)
            ax4 = sns.heatmap(pd.DataFrame(filteredQ), cmap="viridis")
            plt.title("After Filtering - Q")
            plt.xlabel("Sample Index")
            plt.ylabel("Frame Index")
            plt.xticks(np.arange(0, nSample, 100), np.arange(0, nSample, 100))
            plt.yticks(np.arange(0, nSignal, 100), np.arange(0, nSignal, 100))

            plt.savefig(outputPath + fileName + ".png")
            plt.close()

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print("[loopback:" + str(tb.tb_lineno) + "] " + str(ex) + "\n\n")
