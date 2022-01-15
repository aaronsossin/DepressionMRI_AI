import pandas as pd
import numpy as np
import glob
from file_structure import *
import os

dir = "/Users/aaronsossin/Documents/Stanford/depression"


participants_df = pd.read_csv("participants.tsv",sep='\t')

print(participants_df.shape)

controls = participants_df[participants_df.group == "control"]
depressed = participants_df[participants_df.group == "depr"]

print(controls.shape)
print(depressed.shape)

print(participants_df.columns)

dataset = Dataset()

print(dir + "/*")
for folder in glob.glob(dir + "/*"):
    print(folder)
    if "sub" in folder:
        id = folder.split("/")[-1]
        dataset.patient_folders[id] = patient_folder(id=folder.split("/")[-1],path=folder)
        
print(dataset.patient_folders.keys())

dataset.populate_image_arrays()

dataset_matrix = dataset.return_brain_data()

print(dataset_matrix)

