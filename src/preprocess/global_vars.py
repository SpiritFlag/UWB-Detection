from src.global_vars import *

orgSignalPath = dataPathPrefix + "A_signal/"
extractSignalPath = dataPathPrefix + "B_signal/"

onlyTest = False

nTest = int(0.2 * nSignal)
nValidation = int(0.2 * (nSignal - nTest))
nTrain = nSignal - nTest - nValidation
