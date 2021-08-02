from src.global_vars import *
from src.deeplearning.global_vars import *

from src.deeplearning.Train import Train
# from src.preprocess.SetPartition import SetPartition

def main():
    while True:
        try:
            print("")
            print("1. Train")
            # print("2. Test")
            print("")
            print("Input Menu Number: ", end="")
            menu = int(input())
            if menu == 0 or menu > 1:
                raise ValueError(f"invalid menu number: {menu}")
            print("")
            break
        except Exception as ex:
            print(ex)

    if menu == 1:
        instance = Train()
    # elif menu == 2:
    #      instance = SetPartition()

    instance.main()
