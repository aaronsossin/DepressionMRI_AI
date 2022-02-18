import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
from file_structure import *
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
def create_results_folder(save_signature):
    highest = 0
    for folder in glob.glob("Results/*/"):  
        same_signature_folders = [0]
        if save_signature == folder.split("Results/")[1][0:len(save_signature)] and "xx" in folder:
            same_signature_folders.append(int(folder.split("xx")[1].split("xx")[0]))
        highest = max(same_signature_folders) + 1 if max(same_signature_folders) + 1 > highest else highest
    save_folder = "Results/" + save_signature + "xx" + str(highest) + "xx" + str(random.randint(0,1000))
    os.mkdir(save_folder)
    save_folder = save_folder + "/"
    return save_folder

def image_registration(ref_image, off_image, img_to_reg=None, upsample_fac=1): #50
    """ Function to co-register off_image with respect to off_image"""
    shift, error, diffphase = phase_cross_correlation(ref_image, off_image, upsample_factor=upsample_fac)
    if img_to_reg is None:
        img_to_reg = off_image
    reg_image = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img_to_reg), shift)))
    return reg_image

def preprocess(dir,multi=True):
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

    ##### REGISTRATION
    X = X.reshape(int(X.shape[0] / 25), 25, X.shape[1], X.shape[2], X.shape[3])
    # Preprocess for each patient
    for patient in range(X.shape[0]):
        # for each slice location
        for slice_location in range(X.shape[1]):
            ref_image = X[patient,slice_location,:,:,0]
            for time_location in range(1,100):
                X[patient, slice_location,:,:,time_location] = image_registration(ref_image, X[patient, slice_location, :,:,time_location])

    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2],X.shape[3],X.shape[4])
    ######################

    with open("Data/X" + str(multi) + ".npy","wb") as f:
        np.save(f, X)
    
    with open("Data/y" + str(multi) + ".npy","wb") as f:
        np.save(f, y)
    
    return X,y

def standardize(y):
    scaler = StandardScaler()
    y[:,:-1] = scaler.fit_transform(y[:,:-1])
    return y

def plot_y(y,save_folder):
    a = pd.DataFrame(data=y,columns= ["IQ_Raven","Zung_SDS","BDI","MC-SDS","TAS-26","ECR-avoid","ECR-anx","RRS-sum","RRS-reflection","RRS-brooding","RRS-depr","control"])
    print(a)
    a.groupby(["control"],axis=0).mean().plot(kind='bar')
    plt.ylabel("Standardized Score")
    plt.title("Output Scores")
    plt.savefig("Results/prehoc_y.png")