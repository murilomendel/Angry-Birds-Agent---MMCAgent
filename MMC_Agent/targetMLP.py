import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
import os
import cv2
import sys
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input

np.set_printoptions(threshold=np.inf) # Print entire numpy array instruction

if __name__ == "__main__":
    #Setting training dataSet: cd --> path --> filename
    dir_path = os.getcwd()
    file_path = dir_path + "\\dataSet\\prediction\\Target"

    stateI = cv2.imread(file_path + '\\' + sys.argv[1], cv2.IMREAD_GRAYSCALE)
    #stateI = cv2.imread(file_path + '\\predictionImg.png', cv2.IMREAD_GRAYSCALE)

    #print(file_path)
    #print(len(sys.argv))
    #print(str(sys.argv))

    new_model = tf.keras.models.load_model('target_Net')
    Xi = []
    Xi.append(stateI)
    Xi = np.array(Xi).reshape(-1, 2070)
    X = Xi/255

    tgt = []
    print("In Python - Making Network Prediction..")
    prediction = new_model.predict(X)
    
    # Not Reached Zones
    prediction[6] = [[-1]]
    prediction[7] = [[-1]]
    prediction[8] = [[-1]]
    prediction[9] = [[-1]]
    prediction[10] = [[-1]]
    prediction[11] = [[-1]]
    prediction[12] = [[-1]]
    prediction[13] = [[-1]]
    prediction[21] = [[-1]]
    prediction[22] = [[-1]]
    prediction[23] = [[-1]]
    prediction[24] = [[-1]]
    prediction[25] = [[-1]]
    prediction[26] = [[-1]]
    prediction[27] = [[-1]]
    prediction[36] = [[-1]]
    prediction[37] = [[-1]]
    prediction[38] = [[-1]]
    prediction[39] = [[-1]]
    prediction[40] = [[-1]]
    prediction[41] = [[-1]]
    prediction[51] = [[-1]]
    prediction[52] = [[-1]]
    prediction[53] = [[-1]]
    prediction[54] = [[-1]]
    prediction[55] = [[-1]]
    prediction[40] = [[-1]]
    prediction[41] = [[-1]]
    prediction[66] = [[-1]]
    prediction[67] = [[-1]]
    prediction[68] = [[-1]]
    prediction[69] = [[-1]]
    prediction[81] = [[-1]]
    prediction[82] = [[-1]]
    prediction[83] = [[-1]]
    prediction[96] = [[-1]]
    prediction[97] = [[-1]]

    #for i in range(len(prediction)):
    #    print("Zone " + str(i+1) + " = " + str(prediction[i]))
    
    # Extracting Biggest Zone Prediction
    t = np.argmax(prediction)
    tgt.append(int(t+1))
    prediction[t] = [[-1]]
    
    # Extracting Second Biggest Zone Prediction
    t = np.argmax(prediction)
    tgt.append(int(t+1))
	
    print("Best Zones " + str(tgt[0]))
    print("Second Best Zone " + str(tgt[1]))

    flag = randint(1, 5)
    if (flag == 2):
        print(tgt[1])
    else:
        print(tgt[0])