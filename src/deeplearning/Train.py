from src.global_vars import *
from src.deeplearning.global_vars import *
from src.deeplearning.CNN import *

import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Train():
    def __init__(self):
        super(Train, self).__init__()

    def main(self):
        try:
            assert os.path.isdir(signalPath),\
                f"signalPath= {signalPath} "\
                "does not exist.\n"

            cnn = CNN()
            trainSet, trainLabelSet, valSet, valLabelSet = self.readSet()
            hist = cnn.trainModel(trainSet, trainLabelSet, valSet, valLabelSet)
            self.testFnc(cnn)

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[Train.main:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def readSet(self):
        try:
            trainSet = []
            trainLabelSet = []
            valSet = []
            valLabelSet = []

            for fileName in fileNameList:
                for rep in range(1, repetition + 1):
                    set = self.readIQ(fileName + str(rep), "train")
                    trainSet.extend(set)
                    trainLabelSet.extend(
                        self.readLabel(fileName, "train", len(set)))

                    set = self.readIQ(fileName + str(rep), "validation")
                    valSet.extend(set)
                    valLabelSet.extend(
                        self.readLabel(fileName, "validation", len(set)))

            randomIndex = np.arange(len(trainSet))
            np.random.shuffle(randomIndex)

            shuffledTrainSet = []
            shuffledTrainLabelSet = []

            for idx in range(len(randomIndex)):
                shuffledTrainSet.append(trainSet[randomIndex[idx]])
                shuffledTrainLabelSet.append(trainLabelSet[randomIndex[idx]])

            return np.array(shuffledTrainSet), np.array(shuffledTrainLabelSet),\
                np.array(valSet), np.array(valLabelSet)

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[Train.readSet:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def readIQ(self, fileName, postfix):
        try:
            i = np.load(signalPath + fileName + "_Isignal_" + postfix + ".npy")
            q = np.load(signalPath + fileName + "_Qsignal_" + postfix + ".npy")

            # avg = np.mean([i.mean(), q.mean()])
            # std = np.std([i.std(), q.std()])
            # i = (i - avg) / std
            # q = (q - avg) / std

            # avg = i.mean()
            # std = i.std()
            # i = (i - avg) / std
            # avg = q.mean()
            # std = q.std()
            # q = (q - avg) / std

            for idx in range(len(i)):
                avg = i[idx].mean()
                std = i[idx].std()
                i[idx] = (i[idx] - avg) / std
            for idx in range(len(q)):
                avg = q[idx].mean()
                std = q[idx].std()
                q[idx] = (q[idx] - avg) / std

            return np.expand_dims(
                np.swapaxes(np.array([i, q]).transpose(), 0, 1),
                axis=1)[:, :, index_st:index_ed]

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[Train.readIQ:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def readLabel(self, fileName, postfix, size):
        try:
            set = np.zeros((size, classification))

            for x in range(size):
                set[x][int(fileName[-2]) - 1] = 1

            return set

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[Train.readLabel:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def testFnc(self, cnn):
        try:
            tot_cnt = 0
            tot_size = 0
            allPredictSet = []
            allLabelSet = []
            colors = ["b", "g", "r", "c", "m", "y", "k"]

            for fileName in fileNameList:
                print(fileName[:-1], end="\t")
                cur_cnt = 0
                set_size = 0

                for rep in range(1, repetition + 1):
                    testSet = self.readIQ(fileName + str(rep), "test")
                    testLabelSet = self.readLabel(
                        fileName, "test", len(testSet))

                    predictSet = cnn.testModel(testSet)
                    allPredictSet.extend(predictSet)
                    allLabelSet.extend(testLabelSet)

                    cnt = 0
                    for x in range(len(predictSet)):
                        if np.argmax(predictSet[x]) == np.argmax(testLabelSet[x]):
                            cnt += 1

                    print(cnt, end="\t")
                    cur_cnt += cnt
                    set_size += len(testSet)

                print(f"\t{cur_cnt}\t{set_size}\t"\
                    f"{100 * cur_cnt / set_size:.2f}%")
                tot_cnt += cur_cnt
                tot_size += set_size

            print(f"\n\t\tTEST SUCCESS=\t{tot_cnt} /\t{tot_size} "\
                f"({100 * tot_cnt / tot_size:.2f}%)")
            print("\n")

            tsne = TSNE(random_state = 42).fit_transform(allPredictSet)
            for x in range(len(allPredictSet)):
                plt.plot(tsne[x][0], tsne[x][1])
                plt.text(tsne[x][0], tsne[x][1], str(np.argmax(allLabelSet[x])+1),
                    color = colors[np.argmax(allLabelSet[x])],
                    fontdict = {'weight':'bold','size':9})
            plt.savefig("test.png", dpi=300)
            plt.close()

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[Train.testFnc:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")
