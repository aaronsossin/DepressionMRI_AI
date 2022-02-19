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
import numpy as np

img_size = (25,112,112,100)
last_conv_layer_name = "conv3d_3"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        print("last layer shape: ", last_conv_layer_output.shape)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        a = np.array(pred_index)
        print("a.shape", a.shape)
        pred_index = a[0]
        print(pred_index)
        #class_channel = preds[:, pred_index]
        class_channel = preds[pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print(grads.shape)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    
    #pooled_grads = tf.reduce_mean(grads, axis=(0))
    #print(pooled_grads.shape)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation

    last_conv_layer_output = last_conv_layer_output[0]
    #heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = last_conv_layer_output @ grads[..., tf.newaxis]
    print(heatmap.shape)
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

model = keras.models.load_model('Results/letsgoxx1xx45/model')

last_conv_layer_name = ""
for layer in model.layers:
    if "conv" in layer.name:
        last_conv_layer_name = layer.name

print(last_conv_layer_name)
model.layers[-1].activation=None
img = np.load("Data/XTrue.npy")

img_array = img.reshape(72,25,112,112,100)
img_array = img_array[24,:,:,:,:]
img_array = img_array.reshape(1,25,112,112,100)

preds = model.predict(img_array)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name).reshape(12,12,12)

def plot_heatmap(heatmap):
    fig, ax = plt.subplots(6,2)
    axes = ax.ravel()
    for i in range(12):
        axes[i].imshow(heatmap[i,:,:])
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_heatmaps.png")

plot_heatmap(heatmap)

import matplotlib.cm as cm
def save_and_display_gradcam(img_array, heatmap, cam_path="gradcam2.jpg", alpha=0.4):
    # Load the original image
    img = img_array

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    #jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


save_and_display_gradcam(img_array, heatmap)