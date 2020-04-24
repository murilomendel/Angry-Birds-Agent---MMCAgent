import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

# Setting training dataSet: cd --> path --> filename
dir_path = os.getcwd()

training_data = []

# Training Data - Multiple Episodes
print("Collecting data from root...")
file_path = dir_path + "\\dataSet\\trainSet"
filename = file_path + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 2
print("Collecting data from episode 2...")
file_path_2 = dir_path + "\\dataSet\\trainSet\\EP2"
filename_2 = file_path_2 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_2) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_2 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 3
print("Collecting data from episode 3...")
file_path_3 = dir_path + "\\dataSet\\trainSet\\EP3"
filename_3 = file_path_3 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_3) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_3 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 4
print("Collecting data from episode 4...")
file_path_4 = dir_path + "\\dataSet\\trainSet\\EP4"
filename_4 = file_path_4 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_4) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_4 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1
							
# Training Data - Episode 5
print("Collecting data from episode 5..")
file_path_5 = dir_path + "\\dataSet\\trainSet\\EP5"
filename_5 = file_path_5 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_5) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_5 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 6
print("Collecting data from episode 6...")
file_path_6 = dir_path + "\\dataSet\\trainSet\\EP6"
filename_6 = file_path_6 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_6) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_6 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 7
print("Collecting data from episode 7...")
file_path_7 = dir_path + "\\dataSet\\trainSet\\EP7"
filename_7 = file_path_7 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_7) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_7 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 8
print("Collecting data from episode 8...")
file_path_8 = dir_path + "\\dataSet\\trainSet\\EP8"
filename_8 = file_path_8 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_8) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_8 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1
							
# Training Data - Episode 9
print("Collecting data from episode 9...")
file_path_9 = dir_path + "\\dataSet\\trainSet\\EP9"
filename_9 = file_path_9 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_9) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_9 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 10
print("Collecting data from episode 10...")
file_path_10 = dir_path + "\\dataSet\\trainSet\\EP10"
filename_10 = file_path_10 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_10) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_10 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 11
print("Collecting data from episode 11...")
file_path_11 = dir_path + "\\dataSet\\trainSet\\EP11"
filename_11 = file_path_11 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_11) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_11 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 12
print("Collecting data from episode 12...")
file_path_12 = dir_path + "\\dataSet\\trainSet\\EP12"
filename_12 = file_path_12 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_12) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_12 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 13
print("Collecting data from episode 13...")
file_path_13 = dir_path + "\\dataSet\\trainSet\\EP13"
filename_13 = file_path_13 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_13) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_13 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 14
print("Collecting data from episode 14...")
file_path_14 = dir_path + "\\dataSet\\trainSet\\EP14"
filename_14 = file_path_14 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_14) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_14 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 15
print("Collecting data from episode 15...")
file_path_15 = dir_path + "\\dataSet\\trainSet\\EP15"
filename_15 = file_path_15 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_15) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_15 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 16
print("Collecting data from episode 16...")
file_path_16 = dir_path + "\\dataSet\\trainSet\\EP16"
filename_16 = file_path_16 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_16) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_16 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 17

file_path_17 = dir_path + "\\dataSet\\trainSet\\EP17"
filename_17 = file_path_17 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_17) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_17 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward > 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 18
print("Collecting data from episode 18...")
file_path_18 = dir_path + "\\dataSet\\trainSet\\EP18"
filename_18 = file_path_18 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_18) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_18 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1
							
# Training Data - Episode 19
print("Collecting data from episode 19...")
file_path_19 = dir_path + "\\dataSet\\trainSet\\EP19"
filename_19 = file_path_19 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_19) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_19 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 20
print("Collecting data from episode 20...")
file_path_20 = dir_path + "\\dataSet\\trainSet\\EP20"
filename_20 = file_path_20 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_20) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_20 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

# Training Data - Episode 21
print("Collecting data from episode 21...")
file_path_21 = dir_path + "\\dataSet\\trainSet\\EP21"
filename_21 = file_path_21 + "\mlpShot.txt"
ct = 0
ds = 1
for s in range(ds):
    with open(filename_21) as fl:
        for line in fl:
            row = line.split(' ')
            if(int(row[3]) >= 0):
                stateI = cv2.imread(file_path_21 + '\\segmentedResized\\' + row[0], cv2.IMREAD_GRAYSCALE)
                stateT = [int(row[1]), int(row[2])]
                if((stateT[0] > 438) and (stateT[0] < 774)):
                    if((stateT[1] > 198) and (stateT[1] < 367)):
                        #Setting shot zone action
                        xPixel = (((stateT[0] - 119) // 8) - 40)
                        yPixel = ((stateT[1] - 199) // 8)
                        xZone = xPixel // 3
                        yZone = yPixel // 3

                        shotZone = xZone + (yZone*14) + 1
                
                        # extracting each action probability from shot points reached
                        reward = int(row[3])/30000
                        if(reward <= 0.17):
                            reward = -1
                        elif((reward > 0.17) and (reward <= 0.34)):
                            reward = -0.5
                        elif((reward > 0.34) and (reward <= 0.5)):
                            reward = 0
                        elif((reward > 0.5) and (reward <= 0.7)):
                            reward = 0.5
                        elif(reward >= 0.7):
                            reward = 1
                        training_data.append([stateI, shotZone, reward])
                        #if(shotZone == 90):
                            #print(str(row[0]) + " Coordinates(" + str(stateT[0]) + "," + str(stateT[1]) + ") = Zone " + str(shotZone) + " Reward: " + str(reward))
                            #ct = ct+1

print("Shuffling Data...")
random.shuffle(training_data)
print("Data Set Size: " + str(len(training_data)))

Xi = []
z = []
y = []
for im, zo, rw in training_data:

    Xi.append(im)
    z.append(zo)
    #print(rw)
    y.append(rw)

Xi = np.array(Xi).reshape(-1, 23, 90, 1)
Xi = np.array(Xi).reshape(-1, 2070)
z = np.array(z).reshape(-1,1)
y = np.array(y).reshape(-1,1)

X = Xi/255.0

print("Setting Network Archtecture...")
im_input = Input(shape = X.shape[1:])
fc1 = Dense(800)(im_input)
fc2 = Dense(500)(fc1)
fc3 = Dense(98)(fc2)

print("Setting Multiple Outputs...")
output1 = Dense(1)(fc3)
output2 = Dense(1)(fc3)
output3 = Dense(1)(fc3)
output4 = Dense(1)(fc3)
output5 = Dense(1)(fc3)
output6 = Dense(1)(fc3)
output7 = Dense(1)(fc3)
output8 = Dense(1)(fc3)
output9 = Dense(1)(fc3)
output10 = Dense(1)(fc3)
output11 = Dense(1)(fc3)
output12 = Dense(1)(fc3)
output13 = Dense(1)(fc3)
output14 = Dense(1)(fc3)
output15 = Dense(1)(fc3)
output16 = Dense(1)(fc3)
output17 = Dense(1)(fc3)
output18 = Dense(1)(fc3)
output19 = Dense(1)(fc3)
output20 = Dense(1)(fc3)
output21 = Dense(1)(fc3)
output22 = Dense(1)(fc3)
output23 = Dense(1)(fc3)
output24 = Dense(1)(fc3)
output25 = Dense(1)(fc3)
output26 = Dense(1)(fc3)
output27 = Dense(1)(fc3)
output28 = Dense(1)(fc3)
output29 = Dense(1)(fc3)
output30 = Dense(1)(fc3)
output31 = Dense(1)(fc3)
output32 = Dense(1)(fc3)
output33 = Dense(1)(fc3)
output34 = Dense(1)(fc3)
output35 = Dense(1)(fc3)
output36 = Dense(1)(fc3)
output37 = Dense(1)(fc3)
output38 = Dense(1)(fc3)
output39 = Dense(1)(fc3)
output40 = Dense(1)(fc3)
output41 = Dense(1)(fc3)
output42 = Dense(1)(fc3)
output43 = Dense(1)(fc3)
output44 = Dense(1)(fc3)
output45 = Dense(1)(fc3)
output46 = Dense(1)(fc3)
output47 = Dense(1)(fc3)
output48 = Dense(1)(fc3)
output49 = Dense(1)(fc3)
output50 = Dense(1)(fc3)
output51 = Dense(1)(fc3)
output52 = Dense(1)(fc3)
output53 = Dense(1)(fc3)
output54 = Dense(1)(fc3)
output55 = Dense(1)(fc3)
output56 = Dense(1)(fc3)
output57 = Dense(1)(fc3)
output58 = Dense(1)(fc3)
output59 = Dense(1)(fc3)
output60 = Dense(1)(fc3)
output61 = Dense(1)(fc3)
output62 = Dense(1)(fc3)
output63 = Dense(1)(fc3)
output64 = Dense(1)(fc3)
output65 = Dense(1)(fc3)
output66 = Dense(1)(fc3)
output67 = Dense(1)(fc3)
output68 = Dense(1)(fc3)
output69 = Dense(1)(fc3)
output70 = Dense(1)(fc3)
output71 = Dense(1)(fc3)
output72 = Dense(1)(fc3)
output73 = Dense(1)(fc3)
output74 = Dense(1)(fc3)
output75 = Dense(1)(fc3)
output76 = Dense(1)(fc3)
output77 = Dense(1)(fc3)
output78 = Dense(1)(fc3)
output79 = Dense(1)(fc3)
output80 = Dense(1)(fc3)
output81 = Dense(1)(fc3)
output82 = Dense(1)(fc3)
output83 = Dense(1)(fc3)
output84 = Dense(1)(fc3)
output85 = Dense(1)(fc3)
output86 = Dense(1)(fc3)
output87 = Dense(1)(fc3)
output88 = Dense(1)(fc3)
output89 = Dense(1)(fc3)
output90 = Dense(1)(fc3)
output91 = Dense(1)(fc3)
output92 = Dense(1)(fc3)
output93 = Dense(1)(fc3)
output94 = Dense(1)(fc3)
output95 = Dense(1)(fc3)
output96 = Dense(1)(fc3)
output97 = Dense(1)(fc3)
output98 = Dense(1)(fc3)

print("Setting Model...")
model = Model(im_input,[output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13,output14,output15,output16,output17,output18,output19,output20,output21,output22,output23,output24,output25,output26,output27,output28,output29,output30,output31,output32,output33,output34,output35,output36,output37,output38,output39,output40,output41,output42,output43,output44,output45,output46,output47,output48,output49,output50,output51,output52,output53,output54,output55,output56,output57,output58,output59,output60,output61,output62,output63,output64,output65,output66,output67,output68,output69,output70,output71,output72,output73,output74,output75,output76,output77,output78,output79,output80,output81,output82,output83,output84,output85,output86,output87,output88,output89,output90,output91,output92,output93,output94,output95,output96,output97,output98])
#print("Loading Pre-Trained Model...")
#model = tf.keras.models.load_model('target_Net')

print("Setting Multiple Models...")
model1 = Model(im_input,output1)
model2 = Model(im_input,output2)
model3 = Model(im_input,output3)
model4 = Model(im_input,output4)
model5 = Model(im_input,output5)
model6 = Model(im_input,output6)
model7 = Model(im_input,output7)
model8 = Model(im_input,output8)
model9 = Model(im_input,output9)
model10 = Model(im_input,output10)
model11 = Model(im_input,output11)
model12 = Model(im_input,output12)
model13 = Model(im_input,output13)
model14 = Model(im_input,output14)
model15 = Model(im_input,output15)
model16 = Model(im_input,output16)
model17 = Model(im_input,output17)
model18 = Model(im_input,output18)
model19 = Model(im_input,output19)
model20 = Model(im_input,output20)
model21 = Model(im_input,output21)
model22 = Model(im_input,output22)
model23 = Model(im_input,output23)
model24 = Model(im_input,output24)
model25 = Model(im_input,output25)
model26 = Model(im_input,output26)
model27 = Model(im_input,output27)
model28 = Model(im_input,output28)
model29 = Model(im_input,output29)
model30 = Model(im_input,output30)
model31 = Model(im_input,output31)
model32 = Model(im_input,output32)
model33 = Model(im_input,output33)
model34 = Model(im_input,output34)
model35 = Model(im_input,output35)
model36 = Model(im_input,output36)
model37 = Model(im_input,output37)
model38 = Model(im_input,output38)
model39 = Model(im_input,output39)
model40 = Model(im_input,output40)
model41 = Model(im_input,output41)
model42 = Model(im_input,output42)
model43 = Model(im_input,output43)
model44 = Model(im_input,output44)
model45 = Model(im_input,output45)
model46 = Model(im_input,output46)
model47 = Model(im_input,output47)
model48 = Model(im_input,output48)
model49 = Model(im_input,output49)
model50 = Model(im_input,output50)
model51 = Model(im_input,output51)
model52 = Model(im_input,output52)
model53 = Model(im_input,output53)
model54 = Model(im_input,output54)
model55 = Model(im_input,output55)
model56 = Model(im_input,output56)
model57 = Model(im_input,output57)
model58 = Model(im_input,output58)
model59 = Model(im_input,output59)
model60 = Model(im_input,output60)
model61 = Model(im_input,output61)
model62 = Model(im_input,output62)
model63 = Model(im_input,output63)
model64 = Model(im_input,output64)
model65 = Model(im_input,output65)
model66 = Model(im_input,output66)
model67 = Model(im_input,output67)
model68 = Model(im_input,output68)
model69 = Model(im_input,output69)
model70 = Model(im_input,output70)
model71 = Model(im_input,output71)
model72 = Model(im_input,output72)
model73 = Model(im_input,output73)
model74 = Model(im_input,output74)
model75 = Model(im_input,output75)
model76 = Model(im_input,output76)
model77 = Model(im_input,output77)
model78 = Model(im_input,output78)
model79 = Model(im_input,output79)
model80 = Model(im_input,output80)
model81 = Model(im_input,output81)
model82 = Model(im_input,output82)
model83 = Model(im_input,output83)
model84 = Model(im_input,output84)
model85 = Model(im_input,output85)
model86 = Model(im_input,output86)
model87 = Model(im_input,output87)
model88 = Model(im_input,output88)
model89 = Model(im_input,output89)
model90 = Model(im_input,output90)
model91 = Model(im_input,output91)
model92 = Model(im_input,output92)
model93 = Model(im_input,output93)
model94 = Model(im_input,output94)
model95 = Model(im_input,output95)
model96 = Model(im_input,output96)
model97 = Model(im_input,output97)
model98 = Model(im_input,output98)

print("Compilling Model...")
model.compile(loss='mean_squared_error',optimizer='adadelta')

print("Compilling Multiple Models...")
model1.compile(loss='mean_squared_error',optimizer='adadelta')
model2.compile(loss='mean_squared_error',optimizer='adadelta')
model3.compile(loss='mean_squared_error',optimizer='adadelta')
model4.compile(loss='mean_squared_error',optimizer='adadelta')
model5.compile(loss='mean_squared_error',optimizer='adadelta')
model6.compile(loss='mean_squared_error',optimizer='adadelta')
model7.compile(loss='mean_squared_error',optimizer='adadelta')
model8.compile(loss='mean_squared_error',optimizer='adadelta')
model9.compile(loss='mean_squared_error',optimizer='adadelta')
model10.compile(loss='mean_squared_error',optimizer='adadelta')
model11.compile(loss='mean_squared_error',optimizer='adadelta')
model12.compile(loss='mean_squared_error',optimizer='adadelta')
model13.compile(loss='mean_squared_error',optimizer='adadelta')
model14.compile(loss='mean_squared_error',optimizer='adadelta')
model15.compile(loss='mean_squared_error',optimizer='adadelta')
model16.compile(loss='mean_squared_error',optimizer='adadelta')
model17.compile(loss='mean_squared_error',optimizer='adadelta')
model18.compile(loss='mean_squared_error',optimizer='adadelta')
model19.compile(loss='mean_squared_error',optimizer='adadelta')
model20.compile(loss='mean_squared_error',optimizer='adadelta')
model21.compile(loss='mean_squared_error',optimizer='adadelta')
model22.compile(loss='mean_squared_error',optimizer='adadelta')
model23.compile(loss='mean_squared_error',optimizer='adadelta')
model24.compile(loss='mean_squared_error',optimizer='adadelta')
model25.compile(loss='mean_squared_error',optimizer='adadelta')
model26.compile(loss='mean_squared_error',optimizer='adadelta')
model27.compile(loss='mean_squared_error',optimizer='adadelta')
model28.compile(loss='mean_squared_error',optimizer='adadelta')
model29.compile(loss='mean_squared_error',optimizer='adadelta')
print("Compilling Model 30...")
model30.compile(loss='mean_squared_error',optimizer='adadelta')
model31.compile(loss='mean_squared_error',optimizer='adadelta')
model32.compile(loss='mean_squared_error',optimizer='adadelta')
model33.compile(loss='mean_squared_error',optimizer='adadelta')
model34.compile(loss='mean_squared_error',optimizer='adadelta')
model35.compile(loss='mean_squared_error',optimizer='adadelta')
model36.compile(loss='mean_squared_error',optimizer='adadelta')
model37.compile(loss='mean_squared_error',optimizer='adadelta')
model38.compile(loss='mean_squared_error',optimizer='adadelta')
model39.compile(loss='mean_squared_error',optimizer='adadelta')
model40.compile(loss='mean_squared_error',optimizer='adadelta')
model41.compile(loss='mean_squared_error',optimizer='adadelta')
model42.compile(loss='mean_squared_error',optimizer='adadelta')
model43.compile(loss='mean_squared_error',optimizer='adadelta')
model44.compile(loss='mean_squared_error',optimizer='adadelta')
model45.compile(loss='mean_squared_error',optimizer='adadelta')
model46.compile(loss='mean_squared_error',optimizer='adadelta')
model47.compile(loss='mean_squared_error',optimizer='adadelta')
model48.compile(loss='mean_squared_error',optimizer='adadelta')
model49.compile(loss='mean_squared_error',optimizer='adadelta')
model50.compile(loss='mean_squared_error',optimizer='adadelta')
model51.compile(loss='mean_squared_error',optimizer='adadelta')
model52.compile(loss='mean_squared_error',optimizer='adadelta')
model53.compile(loss='mean_squared_error',optimizer='adadelta')
model54.compile(loss='mean_squared_error',optimizer='adadelta')
model55.compile(loss='mean_squared_error',optimizer='adadelta')
model56.compile(loss='mean_squared_error',optimizer='adadelta')
model57.compile(loss='mean_squared_error',optimizer='adadelta')
model58.compile(loss='mean_squared_error',optimizer='adadelta')
model59.compile(loss='mean_squared_error',optimizer='adadelta')
print("Compilling Model 60...")
model60.compile(loss='mean_squared_error',optimizer='adadelta')
model61.compile(loss='mean_squared_error',optimizer='adadelta')
model62.compile(loss='mean_squared_error',optimizer='adadelta')
model63.compile(loss='mean_squared_error',optimizer='adadelta')
model64.compile(loss='mean_squared_error',optimizer='adadelta')
model65.compile(loss='mean_squared_error',optimizer='adadelta')
model66.compile(loss='mean_squared_error',optimizer='adadelta')
model67.compile(loss='mean_squared_error',optimizer='adadelta')
model68.compile(loss='mean_squared_error',optimizer='adadelta')
model69.compile(loss='mean_squared_error',optimizer='adadelta')
model70.compile(loss='mean_squared_error',optimizer='adadelta')
model71.compile(loss='mean_squared_error',optimizer='adadelta')
model72.compile(loss='mean_squared_error',optimizer='adadelta')
model73.compile(loss='mean_squared_error',optimizer='adadelta')
model74.compile(loss='mean_squared_error',optimizer='adadelta')
model75.compile(loss='mean_squared_error',optimizer='adadelta')
model76.compile(loss='mean_squared_error',optimizer='adadelta')
model77.compile(loss='mean_squared_error',optimizer='adadelta')
model78.compile(loss='mean_squared_error',optimizer='adadelta')
model79.compile(loss='mean_squared_error',optimizer='adadelta')
model80.compile(loss='mean_squared_error',optimizer='adadelta')
model81.compile(loss='mean_squared_error',optimizer='adadelta')
model82.compile(loss='mean_squared_error',optimizer='adadelta')
model83.compile(loss='mean_squared_error',optimizer='adadelta')
model84.compile(loss='mean_squared_error',optimizer='adadelta')
model85.compile(loss='mean_squared_error',optimizer='adadelta')
model86.compile(loss='mean_squared_error',optimizer='adadelta')
model87.compile(loss='mean_squared_error',optimizer='adadelta')
model88.compile(loss='mean_squared_error',optimizer='adadelta')
model89.compile(loss='mean_squared_error',optimizer='adadelta')
print("Compilling Model 90...")
model90.compile(loss='mean_squared_error',optimizer='adadelta')
model91.compile(loss='mean_squared_error',optimizer='adadelta')
model92.compile(loss='mean_squared_error',optimizer='adadelta')
model93.compile(loss='mean_squared_error',optimizer='adadelta')
model94.compile(loss='mean_squared_error',optimizer='adadelta')
model95.compile(loss='mean_squared_error',optimizer='adadelta')
model96.compile(loss='mean_squared_error',optimizer='adadelta')
model97.compile(loss='mean_squared_error',optimizer='adadelta')
model98.compile(loss='mean_squared_error',optimizer='adadelta')


print("Training on Batch...")
print("Training Data size = " + str(len(X)))
for i in range(len(training_data)):
    _batch = np.empty(shape=(23, 90, 1))
    _batch = np.array(X[i]).reshape(-1, 2070)
    #print(_batch)
    if(z[i]==1):
        loss = model1.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==2):
        loss = model2.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==3):
        loss = model3.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==4):
        loss = model4.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==5):
        loss = model5.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==6):
        loss = model6.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==7):
        loss = model7.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==8):
        loss = model8.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==9):
        loss = model9.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==10):
        loss = model10.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==11):
        loss = model11.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==12):
        loss = model12.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==13):
        loss = model13.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==14):
        loss = model14.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==15):
        loss = model15.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==16):
        loss = model16.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==17):
        loss = model17.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==18):
        loss = model18.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==19):
        loss = model19.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==20):
        loss = model20.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==21):
        loss = model21.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==22):
        loss = model22.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==23):
        loss = model23.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==24):
        loss = model24.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==25):
        loss = model25.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==26):
        loss = model26.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==27):
        loss = model27.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==28):
        loss = model28.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==29):
        loss = model29.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==30):
        loss = model30.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==31):
        loss = model31.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==32):
        loss = model32.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==33):
        loss = model33.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==34):
        loss = model34.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==35):
        loss = model35.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==36):
        loss = model36.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==37):
        loss = model37.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==38):
        loss = model38.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==39):
        loss = model39.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==40):
        loss = model40.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==41):
        loss = model41.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==42):
        loss = model42.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==43):
        loss = model43.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==44):
        loss = model44.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==45):
        loss = model45.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==46):
        loss = model46.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==47):
        loss = model47.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==48):
        loss = model48.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==49):
        loss = model49.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==50):
        loss = model50.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==51):
        loss = model51.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==52):
        loss = model52.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==53):
        loss = model53.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==54):
        loss = model54.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==55):
        loss = model55.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==56):
        loss = model56.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==57):
        loss = model57.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==58):
        loss = model58.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==59):
        loss = model59.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==60):
        loss = model60.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==61):
        loss = model61.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==62):
        loss = model62.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==63):
        loss = model63.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==64):
        loss = model64.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==65):
        loss = model65.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==66):
        loss = model66.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==67):
        loss = model67.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==68):
        loss = model68.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==69):
        loss = model69.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==70):
        loss = model70.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==71):
        loss = model71.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==72):
        loss = model72.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==73):
        loss = model73.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==74):
        loss = model74.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==75):
        loss = model75.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==76):
        loss = model76.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==77):
        loss = model77.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==78):
        loss = model78.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==79):
        loss = model79.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==80):
        loss = model80.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==81):
        loss = model81.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==82):
        loss = model82.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==83):
        loss = model83.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==84):
        loss = model84.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==85):
        loss = model85.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==86):
        loss = model86.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==87):
        loss = model87.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==88):
        loss = model88.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==89):
        loss = model89.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==90):
        loss = model90.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==91):
        loss = model91.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==92):
        loss = model92.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==93):
        loss = model93.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==94):
        loss = model94.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==95):
        loss = model95.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==96):                
        loss = model96.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==97):
        loss = model97.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))
    if(z[i]==98):
        loss = model98.train_on_batch(_batch,y[i])
        print(" Iteration " + str(i) + " Zone: " + str(z[i]) + " Reward: " + str(y[i]) + " Loss: " + str(loss))

		
print("Saving Model")
model.save('target_Net')
print("Complete!!")
