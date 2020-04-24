import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Setting training dataSet: cd --> path --> filename
dir_path = os.getcwd()
file_path = dir_path + "\\dataSet\\prediction\Target"
print(file_path)
print(len(sys.argv))
print(str(sys.argv))