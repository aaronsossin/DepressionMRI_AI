import pandas as pd
import numpy as np
import glob
from file_structure import *
from keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D, Reshape
from keras.layers import AveragePooling2D, MaxPooling2D,MaxPooling3D,Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import KFold
from keras.preprocessing import image
from keras.utils import layer_utils
import keras
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.neural_network import *
from sklearn.ensemble import *
from sklearn.discriminant_analysis import *
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D
from keras.layers import AveragePooling2D, MaxPooling2D,MaxPooling3D,Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from models import *


########### HYPER-PARAMETERS ####################
dir = "/scratch/users/sosaar/DepressionMRI_AI"
preprocess = False
dimension = 3
epochs = 25
fraction_of_data = 0.10
metrics = [tf.keras.metrics.AUC(),"accuracy"]
learning_rate = 0.0001
loss_function = "binary_crossentropy"
##################################################


if preprocess:
    dataset = Dataset()

    print(dir + "/*")
    for folder in glob.glob(dir + "/*"):
        if "sub" in folder:
            id = folder.split("/")[-1]
            dataset.patient_folders[id] = patient_folder(id=folder.split("/")[-1],path=folder)

    dataset.populate_image_arrays()
    dataset.populate_is_control()

    X, y = dataset.return_brain_data()
    X = np.moveaxis(X,3,1)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2],X.shape[3],X.shape[4])
    
    with open("Data/X.npy","wb") as f:
        np.save(f, X)
    
    with open("Data/y.npy","wb") as f:
        np.save(f, y)

else:
    X = np.load("Data/X.npy")
    y = np.load("Data/y.npy")


if fraction_of_data != 1.0:
    to_keep = np.random.choice(range(0,X.shape[0]),size=int(fraction_of_data * X.shape[0]))
    X = X[to_keep,:,:,:]
    y = y[to_keep]

print(X.shape)
print(y.shape)

kf = KFold(n_splits=4)



scores_dictionary = dict()
for i in range(len(metrics) + 1):
    scores_dictionary[i] = list()

counter = 0
for train_index, test_index in kf.split(X):
    m = model(metrics=metrics,loss_function=loss_function,learning_rate = learning_rate)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    history = m.fit(X_train,y_train,epochs=epochs,validation_split=0.1)
    if counter == 0:
        print_history(history)
    scores = m.evaluate(X_test,y_test)
    for i, s in enumerate(scores):
        scores_dictionary[i].append(s)

    #print(scores)
    counter += 1

print("FINAL SCORES")
print(scores_dictionary)
for x in scores_dictionary:
    scores_dictionary[x] = np.mean(scores_dictionary[x])
print("AVERAGED")
print(scores_dictionary)







