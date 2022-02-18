import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, recall_score, precision_score, balanced_accuracy_score, accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import pickle
import math

def roc_graphs(x_path, y_path, save_folder, save_name = "roc.png"):
    x = np.load(x_path)
    x = np.transpose(x).reshape(int(np.prod(list(x.shape))/12), 12)
    x[:,-1][x[:,-1] >= 0.5] = 1
    x[:,-1] = x[:,-1].astype('int')
    print(x.shape)

    # Load y
    y = np.load(y_path)
    print(y.shape)

def derive_and_average_scores_from_run(x_paths, y_paths, save_folder, save_name="MeanResults.pickle"):

    for i,x_path in enumerate(x_paths):
        if i == 0:
            X = np.load(x_path)
            X = np.transpose(X).reshape(int(np.prod(list(X.shape))/12), 12)
            X[:,-1][X[:,-1] >= 0.5] = 1
            X[:,-1] = X[:,-1].astype('int')
        else:
            x = np.load(x_path)
            x = np.transpose(x).reshape(int(np.prod(list(x.shape))/12), 12)
            x[:,-1][x[:,-1] >= 0.5] = 1
            x[:,-1] = x[:,-1].astype('int')
            X = np.concatenate((X,x),axis=0)
        
    for i,y_path in enumerate(y_paths):
        if i == 0:
            Y = np.load(y_path)
        else:
            Y = np.concatenate((Y,np.load(y_path)),axis=0)
    
    # Save Pred vs. True labels
    df = pd.DataFrame(columns=["Prediction","Ground_Truth"])
    df["Prediction"] = X[:,-1]
    df["Ground_Truth"] = Y[:,-1]
    df["Prediction"] = df["Prediction"].astype("int")
    df["Ground_Truth"] = df["Ground_Truth"].astype("int")
    df.to_csv(save_folder + "labels.csv")
    

    # Count scores
    scores = dict()
    evaluation_metrics = [mean_squared_error, mean_absolute_error, recall_score, precision_score, accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score]
    em_names = ["mse","mae","recall","precision","accuracy","balanced_accuracy","auprc","auroc"]
    metrics = ["IQ_Raven","Zung_SDS","BDI","MC-SDS","TAS-26","ECR-avoid","ECR-anx","RRS-sum","RRS-reflection","RRS-brooding","RRS-depr","Major_Depression"]
    for ix,m in enumerate(metrics):
        scores[m] = dict()

        if m == "Major_Depression":
            # Genereate classification report
            report = classification_report(Y[:,ix], X[:,ix], output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(save_folder + "classification_report.csv")

        for i,em in enumerate(evaluation_metrics):
            if i >= 2 and m != "Major_Depression":
                scores[m][em_names[i]] = float("NaN")
            else:
                if em_names[i] == "auprc":
                    # Weighted auprc
                    scores[m][em_names[i]] = em(Y[:,ix],X[:,ix],average='weighted')
                else:
                    scores[m][em_names[i]] = em(Y[:,ix],X[:,ix])

    with open(save_folder + save_name, 'wb') as handle:
        pickle.dump(scores, handle)
    

    
    return scores

def difference_between_two_score_dictionaries(score_dicts,save_folder = "PermutationImportanceResults/", save_name="SubtractedResults.pickle"):
    sub_dict = dict()
    for m in score_dicts[0]:
        sub_dict[m] = dict()
        for k in score_dicts[0][m]:
            a = score_dicts[0][m].get(k,float("NaN"))
            b = score_dicts[1][m].get(k,float("NaN"))
            if math.isnan(a) or math.isnan(b):
                sub_dict[m][k] = float("NaN")
            else:
                sub_dict[m][k] = a-b #np.nansubtract([d[m].get(k,float("NaN")) for d in score_dicts])
    
    with open(save_folder + save_name, 'wb') as handle:
        pickle.dump(sub_dict, handle)
    
    return sub_dict



"""
BELOW IS DEPRECATED
"""
def try_one_example():
    a = np.load("/scratch/users/sosaar/DepressionMRI_AI/Results/letsgoxx1xx45/p0.npy")
    a = np.transpose(a).reshape(36,12)
    print(a.shape)
    print(a[:,-1])

    binary_a = a[:,-1]
    binary_a[binary_a > 0.5] = 1

    binary_a = binary_a.astype('int')
    print(binary_a)

    b = np.load("/scratch/users/sosaar/DepressionMRI_AI/Results/letsgoxx1xx45/y0.npy")
    print(b.shape)
    print(b[:,-1])

    for i in range(a.shape[1] - 1):
        print(mean_absolute_error(a[:,i],b[:,i]))
    print("--")
    print(recall_score(binary_a,b[:,-1]))
    print(precision_score(binary_a,b[:,-1]))
    print(accuracy_score(binary_a,b[:,-1]))

def average_multiple_kfolds(score_dicts,save_folder = "PermutationImportanceResults/", save_name="MeanResults.pickle"):
    mean_dict = dict()
    for m in score_dicts[0]:
        mean_dict[m] = dict()
        for k in score_dicts[0][m]:
            mean_dict[m][k] = np.nanmean([d[m].get(k,float("NaN")) for d in score_dicts])
    
    with open(save_folder + save_name, 'wb') as handle:
        pickle.dump(mean_dict, handle)
    
    return mean_dict


def derive_scores(y_path,x_path,save_folder="PermutationImportanceResults/",save_name="Results.pickle"):
    print("deriving..")
    # Load x exactly as produced from X.predict
    x = np.load(x_path)
    x = np.transpose(x).reshape(int(np.prod(list(x.shape))/12), 12)
    x[:,-1][x[:,-1] >= 0.5] = 1
    x[:,-1] = x[:,-1].astype('int')
    print(x.shape)

    # Load y
    y = np.load(y_path)
    print(y.shape)

    # Count scores
    scores = dict()
    evaluation_metrics = [mean_squared_error, mean_absolute_error, recall_score, precision_score, accuracy_score, average_precision_score, roc_auc_score]
    em_names = ["mse","mae","recall","precision","accuracy","auprc","auroc"]
    metrics = ["IQ_Raven","Zung_SDS","BDI","MC-SDS","TAS-26","ECR-avoid","ECR-anx","RRS-sum","RRS-reflection","RRS-brooding","RRS-depr","Major_Depression"]
    for ix,m in enumerate(metrics):
        scores[m] = dict()
        for i,em in enumerate(evaluation_metrics):
            if i >= 2 and m != "Major_Depression":
                scores[m][em_names[i]] = float("NaN")
            else:
                scores[m][em_names[i]] = em(y[:,ix],x[:,ix])

    #print(scores)
    with open(save_folder + save_name, 'wb') as handle:
        pickle.dump(scores, handle)
    
    return scores



















# import matplotlib.pyplot as plt
# result = derive_and_average_scores_from_run(["/scratch/users/sosaar/DepressionMRI_AI/Results/letsgo8e-05xx1xx775/p0.npy","/scratch/users/sosaar/DepressionMRI_AI/Results/letsgo8e-05xx1xx775/p1.npy"],["/scratch/users/sosaar/DepressionMRI_AI/Results/letsgo8e-05xx1xx775/y0.npy","/scratch/users/sosaar/DepressionMRI_AI/Results/letsgo8e-05xx1xx775/y1.npy"],"")
# df = pd.DataFrame(result)
# df = df[1:] # omit MSE
# df.iloc[0,:] = df.iloc[0,:] / np.sum(df.iloc[0,:]) # make MAE relative
# df.transpose().plot.bar()
# plt.axhline(y=0.5,linestyle='--',c='black')
# plt.tight_layout()
# plt.savefig("quantitative_scores_all.png")

# scores = average_multiple_kfolds([derive_scores("/scratch/users/sosaar/DepressionMRI_AI/Results/letsgoxx3xx600/y1.npy","/scratch/users/sosaar/DepressionMRI_AI/Results/letsgoxx3xx600/p1.npy")
# ,derive_scores("/scratch/users/sosaar/DepressionMRI_AI/Results/letsgoxx3xx600/y0.npy","/scratch/users/sosaar/DepressionMRI_AI/Results/letsgoxx3xx600/p0.npy")
# ],"")

# import matplotlib.pyplot as plt
# df = pd.DataFrame(scores)

# df = df[1:]
# df.iloc[0,:] = df.iloc[0,:] / np.sum(df.iloc[0,:])
# print(df)
# df.transpose().plot.bar()
# plt.axhline(y=0.5,linestyle='--',c='black')
# plt.savefig("TEST.png")





    
