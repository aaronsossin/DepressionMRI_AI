
from os import path
import pandas as pd
import glob
from nilearn.image import load_img
import nibabel as nib
import numpy as np

participant_info_modified = pd.read_csv("participants_modified.csv")
metrics = ["IQ_Raven","Zung_SDS","BDI","MC-SDS","TAS-26","ECR-avoid","ECR-anx","RRS-sum","RRS-reflection","RRS-brooding","RRS-depr"]
class patient_folder():
    def __init__(self,id,path):
        self.id = id
        self.path = path
        self.image_array = np.zeros((112,112,25,100))
        self.is_control = False

        self.metrics = np.array(participant_info_modified[participant_info_modified["participant_id"] == self.id][metrics].values.tolist()[0])

        

class Dataset():
    def __init__(self):
        self.patient_folders = dict()
        self.participant_info = pd.read_csv("participants.tsv",sep='\t')
    
    def populate_image_arrays(self):

        for pf in self.patient_folders:
            pf_object = self.patient_folders[pf]
            pf_object.image_array = np.array(nib.load(pf_object.path + "/func/" + pf_object.id + "_task-rest_bold.nii.gz").dataobj)
    
    def populate_is_control(self):

        for pf in self.patient_folders:
            pf_object = self.patient_folders[pf]
            pf_object.is_control = True if self.participant_info[self.participant_info["participant_id"] == pf_object.id]["group"].values == "control" else False
    
    def return_brain_data(self):

        images = np.zeros((len(self.patient_folders),112,112,25,100))
        outputs = np.zeros((len(self.patient_folders),12))
        for i,pf in enumerate(self.patient_folders):
            #print(self.patient_folders[pf].image_array.shape)
            images[i,:,:,:,:] = self.patient_folders[pf].image_array
            #outputs.append(np.zeros(25) if self.patient_folders[pf].is_control else np.ones(25))
            outputs[i,:] = np.append(self.patient_folders[pf].metrics, np.array([0]) if  self.patient_folders[pf].is_control else np.array([1]))
        #outputs = np.array(outputs).flatten()
        return images, outputs

    
    
