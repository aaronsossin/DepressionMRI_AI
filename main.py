import pandas as pd
import numpy as np
import glob
from auxiliary import *
import random
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
import gc
from posthoc import *

########### HYPER-PARAMETERS ####################
dir = "/scratch/users/sosaar/DepressionMRI_AI"
preprocess = False
dimension = 4
epochs = 50
fraction_of_data = 1.0
model_version = 5
major_depression_label_only = False
metrics = [tf.keras.metrics.AUC(),"accuracy","Recall","Precision", "mean_squared_error"]
learning_rate = 9e-6
kfolds = 2
loss_function =  ["mean_squared_error"]*11 + ["binary_crossentropy"]
print(loss_function)
multi = True
save_signature = "heavy_vox"
plot_prehoc = False
save_model = True
##################################################
save_signature = save_signature + "_" + str(learning_rate) + "_"
save_signature = save_signature + "d" + str(dimension) + "_"
save_signature = save_signature + "k" + str(kfolds) + "_"
save_signature = save_signature + "m" + str(model_version) + "_"

if dimension == 3:
    fraction_of_data = 0.05

if preprocess and fraction_of_data != 1.0:
    assert False

if preprocess:
    X,y = preprocess(dir,multi)
else:
    X = np.load("Data/X" + str(multi) + ".npy")
    y = np.load("Data/y" + str(multi) + ".npy")

y = standardize(y)

save_folder = create_results_folder(save_signature)

if plot_prehoc:
    plot_y(y, save_folder)

print(X.shape)
print(y.shape)

if dimension == 4:
    X = X.reshape(int(X.shape[0] / 25), 25, X.shape[1], X.shape[2], X.shape[3])
    if model_version == 1:
        chosen_model = model_3d
    elif model_version == 2:
        chosen_model = model_3d_2
    elif model_version == 3:
        chosen_model = model_3d_3
    elif model_version == 4:
        chosen_model = VoxCNN
    else:
        chosen_model = VoxCNN2
else:
    X = X.reshape(int(X.shape[0] / 25), 25, X.shape[1], X.shape[2], X.shape[3])
    X = np.moveaxis(X, 4, 0)
    X = X.reshape(7200, 25, 112, 112)
    y = np.repeat(y, 100,axis=0)
    print(y.shape)
    chosen_model = model

if major_depression_label_only:
    y = y[:,-1]

kf = KFold(random_state=42,n_splits=kfolds)

scores_dictionary = dict()
for i in range(100):
    scores_dictionary[i] = list()
counter = 0
preds = dict()

ys = dict()

test_indices = pd.DataFrame(columns=[str(i) for i in range(kfolds)])

for train_index, test_index in kf.split(X):

    # Initialize new model
    m = chosen_model(metrics=metrics,loss_function=loss_function,learning_rate = learning_rate)

    # Index train and test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if fraction_of_data != 1.0:
        to_keep = np.random.RandomState(42).choice(range(0,X_train.shape[0]),size=int(fraction_of_data * X_train.shape[0]))
        X_train = X_train[to_keep,:,:,:]
        y_train = y_train[to_keep]
        print(np.nanmean(y_train[:,-1] == 1))

        # to_keep = np.random.choice(range(0,X_test.shape[0]),size=int(fraction_of_data * X_test.shape[0]))
        # X_test = X_test[to_keep,:,:,:]
        # y_test = y_test[to_keep]
        print(np.nanmean(y_test[:,-1] == 1))

    #Save test indices
    test_indices[str(counter)] = test_index

    # Train model and record training history
    history = m.fit(X_train,y_train,epochs=epochs)

    # Save model history for inspection
    print_history(history,save_folder,counter, loss_function[0])

    # Save prediction scores for posthoc analysis
    scores = m.evaluate(X_test,y_test)
    pred = m.predict(X_test)
    
    
    for i, s in enumerate(scores):
        scores_dictionary[i].append(s)
    preds[counter] = pred 
    ys[counter] = y_test

    # Save model
    if save_model:
        m.save(save_folder + "model" + str(counter))

    # Housecleaning
    del m
    keras.backend.clear_session()
    gc.collect()

    counter += 1

test_indices.to_csv(save_folder + "test_indices.csv")
print("FINAL SCORES")

# Aggregating across kfolds and saving
for i in range(kfolds):
    preds_df = np.array(preds[i])
    y_tests_df = np.array(ys[i])
    #print(y_tests_df)
    with open(save_folder + 'y' + str(i) + '.npy', 'wb') as f:
        np.save(f, y_tests_df)
    with open(save_folder + 'p' + str(i) + '.npy', 'wb') as f:
        np.save(f, preds_df)


# print(scores_dictionary)
# all_scores = pd.DataFrame(columns=[str(i) for i in range(100)])
# for i in range(len(scores_dictionary)):
#     try:
#         all_scores[i] = scores_dictionary[i]
#     except Exception as e:
#         False
# all_scores.to_csv(save_folder + "scores.csv")

# Combined Plot
scores = pd.DataFrame(columns=["mse","mae","recall","precision","accuracy","auprc","auroc"])

x_paths = [save_folder + 'p' + str(i) + '.npy' for i in range(kfolds)]
y_paths = [save_folder + 'y' + str(i) + '.npy' for i in range(kfolds)]

mean_results_dict = derive_and_average_scores_from_run(x_paths, y_paths,save_folder)
df = pd.DataFrame(mean_results_dict)
df.to_csv(save_folder + "meanscores.csv")
df = df[1:] # omit MSE
df.iloc[0,:] = df.iloc[0,:] / np.sum(df.iloc[0,:]) # make MAE relative
df.transpose().plot.bar()
plt.axhline(y=0.5,linestyle='--',c='black')
plt.title(str(mean_results_dict["Major_Depression"]["auprc"]))
plt.tight_layout()
plt.savefig(save_folder + "quantitative_scores_all.png")

roc_graphs(x_paths, y_paths, save_folder)

regression_graphs(x_paths, y_paths, save_folder)














