
from os import path
import pandas as pd
import glob
from nilearn.image import load_img
import nibabel as nib
import numpy as np

class patient_folder():
    def __init__(self,id,path):
        self.id = id
        self.path = path
        self.image_array = None

class Dataset():
    def __init__(self):
        self.patient_folders = dict()
        self.participant_info = pd.read_csv("participants.tsv",sep='\t')
    
    def populate_image_arrays(self):

        for pf in self.patient_folders:
            pf_object = self.patient_folders[pf]
            # for i in glob.glob(pf_object.path + "/**/*"):
            #     if "task" in i:
            #         pf_object.image_array = nib.load(i)
            #         print(pf_object.image_array.shape)
            pf_object.image_array = load_img(pf_object.path + "/func/" + pf_object.id + "_task-rest_bold.nii.gz")
            #print(pf_object.image_array.shape)
    
    def return_brain_data(self):

        images = np.zeros((len(self.patient_folders),112,112,25,100))
        print(images.shape)
        for i,pf in enumerate(self.patient_folders):
            print(self.patient_folders[pf].image_array.shape)
            images[i,:,:,:,:] = self.patient_folders[pf].image_array
        return images

    
    
