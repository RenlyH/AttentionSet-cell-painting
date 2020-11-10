from tqdm import tqdm

import pandas as pd
import numpy as np
from data_loader import generate_data_set, data_label_split

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import multiprocessing as mp




def mini_noise_signal_cv(size, data, treatment, control, model, cv, verbose, bag_perc=0.5):
    mean_mean_accuracy=[]
    std_mean_accuracy=[]
    mean_pred_score_control=[]
    std_pred_score_control=[]
    mean_pred_score_treatment=[]
    std_pred_score_treatment=[]
    
    pool = mp.Pool(19)
    results = [pool.apply_async(z_score, args=(i/100, size, cv, data, model, 'DMSO', "taxol", verbose)) for i in range(5,96,5)]
    results = [p.get() for p in results]
    dic = {}
    for i in results:
        dic.update(i)
    for i in sorted (dic): 
        mean_mean_accuracy.append(dic[i][0])
        std_mean_accuracy.append(dic[i][1])
        mean_pred_score_control.append(dic[i][2])
        std_pred_score_control.append(dic[i][3])
        mean_pred_score_treatment.append(dic[i][4])
        std_pred_score_treatment.append(dic[i][5])
        
    return (mean_mean_accuracy, std_mean_accuracy, mean_pred_score_control, std_pred_score_control, mean_pred_score_treatment, std_pred_score_treatment)



def z_score(perc, size, splits, data, model, control = 0, treatment = 1, verbose = 0):
    mini_batch = generate_data_set(size, perc, data, treatment, control)
    X, y = data_label_split(mini_batch)
    y = y["compound"]
    
    skf = StratifiedKFold(n_splits = splits)
    pred_score_control = np.array([])
    pred_score_treatment = np.array([])
    mean_accuracy = []
    if type(X) == np.ndarray:
        for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), desc = "K fold CV"):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index , "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgs = model.fit(X_train, y_train)
            pred_score_control = np.append(pred_score_control, lgs.predict_proba(X_test[y_test=="DMSO"])[:,0])
            pred_score_treatment = np.append(pred_score_treatment, lgs.predict_proba(X_test[y_test=="taxol"])[:,1])
            mean_accuracy.append(lgs.score(X_test, y_test))
#         print(y_test, lgs.predict_proba(X_test[y_test==0])[:,0], lgs.predict_proba(X_test[y_test==1])[:,1])
    elif type(X) == pd.core.frame.DataFrame:
         for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), desc = "K fold CV"):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index , "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lgs = model.fit(X_train, y_train)
            pred_score_control = np.append(pred_score_control, lgs.predict_proba(X_test[y_test=="DMSO"])[:,1])
            pred_score_treatment = np.append(pred_score_treatment, lgs.predict_proba(X_test[y_test=="taxol"])[:,1])
            mean_accuracy.append(lgs.score(X_test, y_test))
    return {perc: [np.mean(mean_accuracy), np.std(mean_accuracy),
                   np.mean(pred_score_control), np.std(pred_score_control), 
                   np.mean(pred_score_treatment), np.std(pred_score_treatment)]}


def train(size, data, model, verbose, bag_perc = 0.5):
    result1 = mini_noise_signal_cv(size, data, "taxol", "DMSO", model, 10, verbose, bag_perc)
    (mean_mean_accuracy, std_mean_accuracy,
     mean_pred_score_control, std_pred_score_control,
     mean_pred_score_treatment, std_pred_score_treatment) = result1
    
    mean_accuracy = np.array(mean_mean_accuracy)
    std_accuracy = np.array(std_mean_accuracy)
    mean_pred_score_control = np.array(mean_pred_score_control)
    std_pred_score_control = np.array(std_pred_score_control)
    mean_pred_score_treatment = np.array(mean_pred_score_treatment)
    std_pred_score_treatment = np.array(std_pred_score_treatment)
    
    z_score = 1-3*(std_pred_score_control+std_pred_score_treatment)/(np.abs(mean_pred_score_control-mean_pred_score_treatment))
    
    model_name = str(model).split("(")[0]
    feature_size = len(data_label_split(data)[0].columns)
    
    plt.figure(figsize=(14, 5))
    ax = plt.subplot(1,2,1)
    plt.plot([i/100 for i in range(5,96,5)], mean_accuracy, '-')
    plt.fill_between([i/100 for i in range(5,96,5)], 
                     (mean_accuracy)-std_accuracy, mean_accuracy+std_accuracy, alpha=0.2)
    ax.set_title("Accuracy score of %s, %s cells, %s features"%(model_name, 500, feature_size))

    ax = plt.subplot(1,2,2)
    plt.plot([i/100 for i in range(5,96,5)], z_score, '-')
    ax.set_title("Z_score of %s, %s cells, %s features"%(model_name, 500, feature_size))

    plt.savefig("%s_sample%s_feature%s.png" %(model_name, size, feature_size))


    
if __name__=='__main__':
    drop_NA_data=pd.read_csv("moa_data_drop_NA.csv", index_col=0)
    print("data loaded")
    sf_drop_NA_data = drop_NA_data[["compound", "concentration",
                                "moa", "row ID", "Iteration (#2)", "COND",
                               "AreaShape_Area_Nuclei"]]
    lgr = LogisticRegression(max_iter = 10000, solver = "saga", n_jobs = -1)
#     lgr = MLPClassifier(alpha=1, max_iter=1000)
    train(10000, sf_drop_NA_data, lgr, 0)
#     train(10000, sf_drop_NA_data, lgr, 0)