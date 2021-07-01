#cuda_device_id = "-1"
#cuda_device_id = "0"
cuda_device_id = "1"

dataPath = "misc/data/"
expNum = 1

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
