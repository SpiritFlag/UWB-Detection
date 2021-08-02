#cuda_device_id = "-1"
#cuda_device_id = "0"
cuda_device_id = "1"

colors = ["b", "g", "r", "c", "m", "y", "k"]

dataPath = "misc/data/"
expNum = 4

if expNum < 10:
    dataPathPrefix = dataPath + "exp0" + str(expNum) + "_"
else:
    dataPathPrefix = dataPath + "exp" + str(expNum) + "_"

if expNum == 1:
    fileNameListAll = []

    for x in range(1, 5):
        fileNameListAll.append(str(x))

    # fileNameList = ["1"]
    fileNameList = fileNameListAll

    nPreSignal = 50
    nSignal = 500
    nSample = 1016

elif expNum == 2:
    fileNameListAll = []

    for x in range(1, 4):
        for a in range(1, 5):
        # for a in range(4, 5):
            for b in range(1, 6):
            # for b in range(1, 2):
                fileNameListAll.append(str(x) + "_" + str(a) + "_" + str(b))

    # fileNameList = ["1_1_1"]
    fileNameList = fileNameListAll

    nPreSignal = 20
    nSignal = 100
    nSample = 1016

elif expNum == 3:
    fileNameListAll = []
    classification = 4
    repetition = 4

    for a in ["a1"]:
        for b in range(1, classification + 1):
            fileNameListAll.append(a + "_" + str(b) + "_")

    # fileNameList = ["1_1_1"]
    fileNameList = fileNameListAll

    nPreSignal = 100
    nSignal = 500
    nSample = 1016

elif expNum == 4:
    fileNameListAll = []
    classification = 4
    repetition = 1

    for a in ["acryl", "glass", "steel", "wood"]:
            fileNameListAll.append(a + "_")

    fileNameList = fileNameListAll

    nPreSignal = 90
    nSignal = 2800
    nSample = 197
