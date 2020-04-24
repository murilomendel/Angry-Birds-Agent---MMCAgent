import os # Use of operating system
import sys
from stat import *
from random import randint

if __name__ == "__main__":
    flag = randint(0, 9)
    print("In Python - Number 0 to 9 selected: " + str(flag))
    if (flag % 2) == 1:
        print("2")
    else:
        print("1")	