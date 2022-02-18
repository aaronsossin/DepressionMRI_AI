import pandas as pd
import numpy as np

# Import Participants
df = pd.read_csv("participants.tsv",sep='\t')
print(df)

# Drop some columns
df = df.drop(columns=["age","gender","ICD-10", "group"])

# Identify columns that are too NaN filled and Drop
columns_to_drop = list()
for i in df.columns[1:]:
    if df[~np.isnan(df[i])].shape[0] < df.shape[0] / 2:
        columns_to_drop.append(i)
df = df.drop(columns=columns_to_drop)

# What's left?
print("Remaining columns")
print(df.shape)
print(df.columns)

# Fill remaining nan with mean imputation
for i in df.columns[1:]:
    df[i].fillna(value=np.nanmean(df[i]),inplace=True)

# Save file
df.to_csv("participants_modified.csv")


