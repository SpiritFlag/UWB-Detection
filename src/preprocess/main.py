from src.global_vars import *
from src.preprocess.global_vars import *

from src.preprocess.SignalExtraction import SignalExtraction
from src.preprocess.SetPartition import SetPartition

def main():
    while True:
        try:
            print("")
            print("1. Signal Extraction")
            print("2. Set Partition")
            print("")
            print("Input Menu Number: ", end="")
            menu = int(input())
            if menu == 0 or menu > 2:
                raise ValueError(f"invalid menu number: {menu}")
            print("")
            break
        except Exception as ex:
            print(ex)

    if menu == 1:
        instance = SignalExtraction()
    elif menu == 2:
         instance = SetPartition()

    instance.main()
