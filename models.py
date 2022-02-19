
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D, Reshape
from keras.layers import AveragePooling2D, MaxPooling2D,MaxPooling3D,Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import KFold
from keras.preprocessing import image
from keras.utils import layer_utils
import keras
import matplotlib.pyplot as plt
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
from keras.utils.vis_utils import plot_model
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D
from keras.layers import AveragePooling2D, MaxPooling2D,MaxPooling3D,Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam


def model(learning_rate=0.0001, loss_function="binary_crossentropy",metrics=["accuracy"]):
    inputs = Input((25,112,112))
    inputs2 = Reshape((25,112,112,1))(inputs)
    #inputs2 = Reshape((112,112,100,1))(inputs)
    c1 = Conv3D(10, (6,8,8), strides=(1, 2,2), activation='relu')(inputs2)
    c2 = MaxPooling3D((2,2,2))(c1)
    c3 = Conv3D(10, (5,6,6), strides=(2,3,3), activation='relu')(c2)
    layer = MaxPooling3D((2,2,2))(c3)
    #c5 = Conv3D(30, (5,5,3), strides=(2,2,1),activation='relu')(c4)
    c6 = Flatten()(layer)

    o1 = Dense(1,activation='tanh')(c6)
    o1 = Flatten()(o1)
    o2 = Dense(1,activation='tanh')(c6)
    o2 = Flatten()(o2)
    o3 = Dense(1,activation='tanh')(c6)
    o3 = Flatten()(o3)
    o4 = Dense(1,activation='tanh')(c6)
    o4 = Flatten()(o4)
    o5 = Dense(1,activation='tanh')(c6)
    o5 = Flatten()(o5)
    o6 = Dense(1,activation='tanh')(c6)
    o6 = Flatten()(o6)
    o7 = Dense(1,activation='tanh')(c6)
    o7 = Flatten()(o7)
    o8 = Dense(1,activation='tanh')(c6)
    o8 = Flatten()(o8)
    o9 = Dense(1,activation='tanh')(c6)
    o9 = Flatten()(o9)
    o10 = Dense(1,activation='tanh')(c6)
    o10 = Flatten()(o10)
    o11 = Dense(1,activation='tanh')(c6)
    o11 = Flatten()(o11)
    o12 = Dense(1,activation='sigmoid')(c6)
    o12 = Flatten()(o12)
    model = Model(inputs=inputs,outputs=[o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12])
    model.summary()

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss_function)#, metrics = metrics)
    return model

def model_3d_3(learning_rate=0.0001, loss_function="binary_crossentropy",metrics=["accuracy"]):
    inputs = Input((25,112,112,100))
    
    layer = Conv3D(30, (4,8,8), strides=(1,2,2), activation='relu')(inputs)
    layer = MaxPooling3D((2,2,2))(layer)
    layer = Conv3D(20, (4,4,4), strides=(2,2,2), activation='relu')(layer)
    layer = MaxPooling3D(((2,2,2)))(layer)
    layer = Conv3D(10, (1,3,3))(layer)
    c6 = Flatten()(layer)

    o1 = Dense(1,activation='tanh')(c6)
    o1 = Flatten()(o1)
    o2 = Dense(1,activation='tanh')(c6)
    o2 = Flatten()(o2)
    o3 = Dense(1,activation='tanh')(c6)
    o3 = Flatten()(o3)
    o4 = Dense(1,activation='tanh')(c6)
    o4 = Flatten()(o4)
    o5 = Dense(1,activation='tanh')(c6)
    o5 = Flatten()(o5)
    o6 = Dense(1,activation='tanh')(c6)
    o6 = Flatten()(o6)
    o7 = Dense(1,activation='tanh')(c6)
    o7 = Flatten()(o7)
    o8 = Dense(1,activation='tanh')(c6)
    o8 = Flatten()(o8)
    o9 = Dense(1,activation='tanh')(c6)
    o9 = Flatten()(o9)
    o10 = Dense(1,activation='tanh')(c6)
    o10 = Flatten()(o10)
    o11 = Dense(1,activation='tanh')(c6)
    o11 = Flatten()(o11)
    o12 = Dense(1,activation='sigmoid')(c6)
    o12 = Flatten()(o12)
    model = Model(inputs=inputs,outputs=[o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12])
    model.summary()

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss_function)#, metrics = metrics)
    return model

def model_3d_2(learning_rate=0.0001, loss_function="binary_crossentropy",metrics=["accuracy"]):
    inputs = Input((25,112,112,100))
    
    layer = Conv3D(30, (5,5,5), strides=(2,2,2), activation='relu')(inputs)
    layer = MaxPooling3D((2,2,2))(layer)
    layer = Conv3D(20, (4,4,4), strides=(2,2,2), activation='relu')(layer)
    layer = MaxPooling3D(((1,2,2)))(layer)
    c6 = Flatten()(layer)

    o1 = Dense(1,activation='tanh')(c6)
    o1 = Flatten()(o1)
    o2 = Dense(1,activation='tanh')(c6)
    o2 = Flatten()(o2)
    o3 = Dense(1,activation='tanh')(c6)
    o3 = Flatten()(o3)
    o4 = Dense(1,activation='tanh')(c6)
    o4 = Flatten()(o4)
    o5 = Dense(1,activation='tanh')(c6)
    o5 = Flatten()(o5)
    o6 = Dense(1,activation='tanh')(c6)
    o6 = Flatten()(o6)
    o7 = Dense(1,activation='tanh')(c6)
    o7 = Flatten()(o7)
    o8 = Dense(1,activation='tanh')(c6)
    o8 = Flatten()(o8)
    o9 = Dense(1,activation='tanh')(c6)
    o9 = Flatten()(o9)
    o10 = Dense(1,activation='tanh')(c6)
    o10 = Flatten()(o10)
    o11 = Dense(1,activation='tanh')(c6)
    o11 = Flatten()(o11)
    o12 = Dense(1,activation='sigmoid')(c6)
    o12 = Flatten()(o12)
    model = Model(inputs=inputs,outputs=[o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12])
    model.summary()

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss_function)#, metrics = metrics)
    return model
def model_3d(learning_rate=0.0001, loss_function="binary_crossentropy",metrics=["accuracy"]):
    inputs = Input((25,112,112,100))
    
    layer = Conv3D(20, (5,5,5), strides=(2,2,2), activation='relu')(inputs)
    layer = MaxPooling3D((2,2,2))(layer)
    layer = Conv3D(10, (4,4,4), strides=(2,2,2), activation='relu')(layer)
    layer = MaxPooling3D(((1,2,2)))(layer)
    c6 = Flatten()(layer)

    o1 = Dense(1,activation='tanh')(c6)
    o1 = Flatten()(o1)
    o2 = Dense(1,activation='tanh')(c6)
    o2 = Flatten()(o2)
    o3 = Dense(1,activation='tanh')(c6)
    o3 = Flatten()(o3)
    o4 = Dense(1,activation='tanh')(c6)
    o4 = Flatten()(o4)
    o5 = Dense(1,activation='tanh')(c6)
    o5 = Flatten()(o5)
    o6 = Dense(1,activation='tanh')(c6)
    o6 = Flatten()(o6)
    o7 = Dense(1,activation='tanh')(c6)
    o7 = Flatten()(o7)
    o8 = Dense(1,activation='tanh')(c6)
    o8 = Flatten()(o8)
    o9 = Dense(1,activation='tanh')(c6)
    o9 = Flatten()(o9)
    o10 = Dense(1,activation='tanh')(c6)
    o10 = Flatten()(o10)
    o11 = Dense(1,activation='tanh')(c6)
    o11 = Flatten()(o11)
    o12 = Dense(1,activation='sigmoid')(c6)
    o12 = Flatten()(o12)
    model = Model(inputs=inputs,outputs=[o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12])
    model.summary()

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss_function)#, metrics = metrics)
    return model

def print_history(history,save_folder,k):
    metrics = ["IQ_Raven","Zung_SDS","BDI","MC-SDS","TAS-26","ECR-avoid","ECR-anx","RRS-sum","RRS-reflection","RRS-brooding","RRS-depr","Major_Depression"]

    print(list(history.history.keys()))
    keys = list(history.history.keys())

    # Get specific names of losses
    num=0
    for i in keys:
        if "flatten" in i:
            num = int(i.split("flatten_")[1].split("_loss")[0])
            break
    print(num)
    fig, ax = plt.subplots()
    for i in range(0,11):
        ax.plot(history.history['flatten_' + str(num+i) + '_loss'],label=str(metrics[i]))
    ax.set_ylabel("mean_squared_error")
    ax.legend(bbox_to_anchor=(1.15,1), loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(history.history['flatten_' + str(num+11) + '_loss'],label=str(metrics[11]),c='black',linestyle='--')
    ax2.set_ylabel("binary_crossentropy")
    ax2.legend(bbox_to_anchor=(1.10,0), loc="upper left")

    plt.title('Model Loss during Training')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + "history" + str(k) + ".png")
    plt.close()

