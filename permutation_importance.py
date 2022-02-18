#https://keras.io/examples/vision/grad_cam/

from IPython.display import Image, display
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, recall_score, precision_score, accuracy_score
import numpy as np
import os
from posthoc import *

save_signature = "please_1e-06_d4_k2_m1_xx1xx677"


if not os.path.isdir("PermutationImportanceResults/" + save_signature + "/"):
    os.mkdir("PermutationImportanceResults/" + save_signature + "/")

model = keras.models.load_model("saved_results/" + save_signature + "/model0")
img = np.load("Data/XTrue.npy")
y = np.load("Data/yTrue.npy")

#test_indices = pd.read_csv("saved_results/" + save_signature + "/test_indices.csv")

X_test = img.reshape(72,25,112,112,100)
X_test = X_test[36:,:,:,:,:]
y_test = y[36:]

with open("PermutationImportanceResults/" + save_signature + "/y_test.npy", 'wb') as f:
            np.save(f, y_test)

def permutation_time_axis(X_test,size=50):
    for i in [0,50]:
        print(i)
        X = X_test.copy()
        X[:,:,:,:,i:i+size] = np.random.normal(np.mean(X[:,:,:,:,i:i+size]),np.std(X[:,:,:,:,i:i+size]),size=[X.shape[0],X.shape[1],X.shape[2],X.shape[3],size])
        y_pred = model.predict(X)
        
        with open("PermutationImportanceResults/" + save_signature + "/timeaxis" + str(i) + '.npy', 'wb') as f:
            np.save(f, y_pred)

def permutation_spatial(X_test,size=56):
    for i,j in [(0,56),(0,0),(56,0),(56,56)]:
        X = X_test.copy()
        np.random.shuffle(X[:,:,i:i+size,j:j+size,:])
        X[:,:,i:i+size,j:j+size,:] = np.random.normal(np.mean(X[:,:,i:i+size,j:j+size,:]),np.std(X[:,:,i:i+size,j:j+size,:]),size=[X.shape[0],X.shape[1],size,size,100])
        y_pred = model.predict(X)
        with open("PermutationImportanceResults/" + save_signature + "/spatialaxis" + str(i) + ":" + str(j) + '.npy', 'wb') as f:
            np.save(f, y_pred)

#permutation_time_axis(X_test)
#permutation_spatial(X_test)

scores = derive_scores("PermutationImportanceResults/" + save_signature + "/y_test.npy","PermutationImportanceResults/" + save_signature + "/spatialaxis56:56.npy")
scores2 = derive_scores("PermutationImportanceResults/" + save_signature + "/y_test.npy", "saved_results/" + save_signature + "/p0.npy")
print(scores)
diff = difference_between_two_score_dictionaries([scores, scores2], "")
print(diff)

df = pd.DataFrame(diff)
df = df[2:] # omit MSE
#df.iloc[0,:] = df.iloc[0,:] / np.sum(df.iloc[0,:]) # make MAE relative
df.transpose().plot.bar()
plt.axhline(y=0.5,linestyle='--',c='black')
plt.tight_layout()
plt.savefig("permutation.png")







