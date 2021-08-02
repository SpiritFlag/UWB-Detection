from src.global_vars import *
from src.preprocess.global_vars import *

import os
import sys
import numpy as np
from tqdm import tqdm

class SetPartition():
    def __init__(self):
        super(SetPartition, self).__init__()

    def main(self):
        try:
            assert os.path.isdir(extractSignalPath),\
                f"orgSignalPath= {extractSignalPath} "\
                "does not exist.\n"

            if onlyTest is True:
                index = [[], [], np.arange(nSignal)]
            else:
                index = self.getIndex()

            for fileName in fileNameList:
                for rep in range(1, repetition + 1):
                    # typeList = ["Isignal", "Qsignal"]
                    typeList = ["signal"]
                    postfixList = ["train", "validation", "test"]
                    path = extractSignalPath + fileName + str(rep) + "_"

                    for type in typeList:
                        set = np.load(path + type + ".npy")

                        for x in range(len(postfixList)):
                            postfix = "_" + postfixList[x]
                            np.save(path + type + postfix, set[index[x]])

            print("Set Partition Complete!", end="\n\n")

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[SetPartition:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def getIndex(self):
        try:
            assert os.path.isdir(extractSignalPath),\
                f"orgSignalPath= {extractSignalPath} "\
                "does not exist.\n"

            if onlyTest is True:
                index = [[], [], np.arange(nSignal)]

            else:
                while True:
                    try:
                        print("Input Index Number(0 to create new index): ",
                            end="")
                        menu = int(input())

                        if menu == 0:
                            index = self.makeIndex()
                            while True:
                                try:
                                    print("Input Index Number to Save: ",
                                        end="")
                                    saveIdx = int(input())
                                    path = extractSignalPath + "index_" + \
                                        str(saveIdx) + "_"
                                    assert not os.path.isfile(path + \
                                        "train.npy"),\
                                        f"index file {saveIdx} does exist.\n"
                                    np.save(path + "train", index[0])
                                    np.save(path + "validation", index[1])
                                    np.save(path + "test", index[2])
                                    break

                                except Exception as ex:
                                    print(ex)

                        else:
                            path = extractSignalPath + "index_" + str(menu) + \
                                "_"
                            assert os.path.isfile(path + "train.npy"),\
                                f"index file {menu} does not exist.\n"
                            index = []
                            index.append(np.load(path + "train.npy"))
                            index.append(np.load(path + "validation.npy"))
                            index.append(np.load(path + "test.npy"))
                        break

                    except Exception as ex:
                        _, _, tb = sys.exc_info()
                        print("[SetPartition:" + str(tb.tb_lineno) + "] " + \
                            str(ex))

            return index

        except Exception as ex:
            _, _, tb = sys.exc_info()
            print("[SetPartition:" + str(tb.tb_lineno) + "] " + str(ex) +
                "\n\n")

    def makeIndex(self):
        index = np.arange(nSignal)
        np.random.shuffle(index)

        shuffledIndex = []
        shuffledIndex.append(index[:nTrain])
        shuffledIndex.append(index[nTrain:nTrain+nValidation])
        shuffledIndex.append(index[nTrain+nValidation:])

        return shuffledIndex
