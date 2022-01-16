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
    inputs = Input((112,112,100))
    inputs2 = Reshape((112,112,100,1))(inputs)
    #inputs2 = Reshape((112,112,100,1))(inputs)
    c1 = Conv3D(20, (1, 1, 23), strides=(1, 1, 8), activation='relu')(inputs2)
    c2 = Conv3D(20, (5,5,1), strides=(2,2,1), activation='relu')(c1)
    c3 = Conv3D(20, (5,5,3), strides=(2,2,1),activation='relu')(c2)
    #c4 = Conv3D(5, (5,5,2), strides=(2,2,1),activation='relu')(c3)
    #c3 = Conv3D(20, (5, 5, 5), strides=(2,2,2),activation='relu')(c2)
    #
    #c2 = Conv3D(10, (1,1,5),strides=(1,1,2),activation='relu')(c1)
    #c3 = Conv3D(50, (5,5,5), strides=(2,2,2),activation='relu')(c2)
    #c4 = Conv3D(10, (2,2,2), strides=(1,1,1), activation='relu')(c3)
    c5 = Flatten()(c3)
    output = Dense(1,activation='sigmoid')(c5)
    model = Model(inputs,output)
    model.summary()

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss_function, metrics = metrics)
    #plot_model(model, to_file='Results/model_plot.png', show_shapes=True, show_layer_names=False)
    #tf.keras.utils.plot_model(model, to_file='model_unet.png', show_shapes=True)
    return model

def print_history(history):
    print(history.history)
    print(history.history['loss'])
    print(history.history.keys())
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.title('Model Loss during Training')
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("Results/history.png")
    plt.close()

model()