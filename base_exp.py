import multiprocessing as mp
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data as D
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import datetime
from data_loader import (data_label_split, 
                         data_standardization, 
                         generate_data_set, 
                         normalize_by_group, 
                         use_nuclei_feature, 
                         use_nuclei_gran_feature,
                         dmso_taxol_ProfileBag)

import warnings
warnings.filterwarnings("ignore")
"""
Conduct 19 different ML tasks on predicting treatment against control at cell-level. Percent of signal cells and noise ranges from 5% to 95%. At each percent, K-fold cross-validation is applied to measure the accuracy and Z' of each model.
"""

def set_args():
    parser = argparse.ArgumentParser(description="Cell mixture bags experiment baseline")
    parser.add_argument("--bagsize", type=int, default=50000, help="bagsize")
    parser.add_argument("--use_nuclei", action="store_true", default=False, help="use nuclei features only")
    parser.add_argument("--use_nuclei_gran", action="store_true", default=False, help="use nuclei granularity features only")
    parser.add_argument("--data_path", type=str, default='moa_data_drop_NA.csv', help="data path")
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        metavar="k",
        help="Total splits for K-fold cross validation",
    )
    parser.add_argument(
        "--mean_bag_length", type=int, default=150, metavar="ML", help="average bag length"
    )
    parser.add_argument(
        "--var_bag_length",
        type=int,
        default=10,
        metavar="VL",
        help="variance of bag length",
    )
    parser.add_argument(
        "--num_bags_train",
        type=int,
        default=60,
        metavar="NTrain",
        help="number of batches in dataset",
    )
    parser.add_argument('--bag_predict',default=False,action='store_true')
    parser.add_argument("--batch_size", type=int, default=20, help="Number of batches")
    parser.add_argument("--output_dir", type=str, default='result/')
    return parser.parse_args()


def mini_noise_signal_cv(
    size: int,
    data: pd.DataFrame,
    treatment: str,
    control: str,
    model,
    cv: int,
    verbose: int,
    bag_perc: float = 0.5,
) -> tuple:
    mean_mean_accuracy = []
    std_mean_accuracy = []
    mean_pred_score_control = []
    std_pred_score_control = []
    mean_pred_score_treatment = []
    std_pred_score_treatment = []
    for i in tqdm(range(5, 96, 5)):
        mini_batch = generate_data_set(
            size, i / 100, data, treatment, control, bag_perc
        )
        X, y = data_label_split(mini_batch)

        # encode string class into numerical class, 0 for control, 1 for treatment
        y = y["compound"]  # .map({treatment:1, control:0})

        mean_accuracy, pred_score_control, pred_score_treatment = kfold_train(
            cv, X, y, model, "DMSO", "taxol", verbose=verbose
        )
        mean_mean_accuracy.append(np.mean(mean_accuracy))
        std_mean_accuracy.append(np.std(mean_accuracy))
        mean_pred_score_control.append(np.mean(pred_score_control))
        std_pred_score_control.append(np.std(pred_score_control))
        mean_pred_score_treatment.append(np.mean(pred_score_treatment))
        std_pred_score_treatment.append(np.std(pred_score_treatment))
    return (
        mean_mean_accuracy,
        std_mean_accuracy,
        mean_pred_score_control,
        std_pred_score_control,
        mean_pred_score_treatment,
        std_pred_score_treatment,
    )


def kfold_train(splits, X, y, model, control=0, treatment=1, verbose=0):
    skf = StratifiedKFold(n_splits=splits)
    pred_score_control = np.array([])
    pred_score_treatment = np.array([])
    mean_accuracy = []
    if type(X) == np.ndarray:
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgs = model.fit(X_train, y_train)
            pred_score_control = np.append(
                pred_score_control, lgs.predict_proba(X_test[y_test == "DMSO"])[:, 0]
            )
            pred_score_treatment = np.append(
                pred_score_treatment, lgs.predict_proba(X_test[y_test == "taxol"])[:, 1]
            )
            mean_accuracy.append(lgs.score(X_test, y_test))
    #         print(y_test, lgs.predict_proba(X_test[y_test==0])[:,0], lgs.predict_proba(X_test[y_test==1])[:,1])
    elif type(X) == pd.core.frame.DataFrame:
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lgs = model.fit(X_train, y_train)
            pred_score_control = np.append(
                pred_score_control, lgs.predict_proba(X_test[y_test == "DMSO"])[:, 1]
            )
            pred_score_treatment = np.append(
                pred_score_treatment, lgs.predict_proba(X_test[y_test == "taxol"])[:, 1]
            )
            mean_accuracy.append(lgs.score(X_test, y_test))
    return mean_accuracy, pred_score_control, pred_score_treatment


"""
Multi-processing version of noise-signal-CV
"""


def multi_mini_noise_signal_cv(
    args: int,
    data: pd.DataFrame,
    treatment: str,
    control: str,
    model,
    verbose: int,
    bag_perc: float = 0.5,
) -> tuple:
    mean_mean_accuracy = []
    std_mean_accuracy = []
    mean_pred_score_control = []
    std_pred_score_control = []
    mean_pred_score_treatment = []
    std_pred_score_treatment = []

    pool = mp.Pool(19)
    if args.bag_predict:
        foo = multi_kfold_train_bag
    else:
        foo = multi_kfold_train
        
    results = [
        pool.apply_async(
            foo,
            args=(i / 100, args, data, model, "DMSO", "taxol", verbose),
        )
        for i in range(5, 96, 5)
    ]
    results = [p.get() for p in results]
    dic = {}
    for i in results:
        dic.update(i)
    for i in sorted(dic):
        mean_mean_accuracy.append(dic[i][0])
        std_mean_accuracy.append(dic[i][1])
        mean_pred_score_control.append(dic[i][2])
        std_pred_score_control.append(dic[i][3])
        mean_pred_score_treatment.append(dic[i][4])
        std_pred_score_treatment.append(dic[i][5])

    return dic  # (mean_mean_accuracy, std_mean_accuracy, mean_pred_score_control, std_pred_score_control, mean_pred_score_treatment, std_pred_score_treatment)


def multi_kfold_train(perc, args, data, model, control=0, treatment=1, verbose=0):
    
    mini_batch = generate_data_set(args.bagsize, perc, data, treatment, control)
    X, y = data_label_split(mini_batch)
    y = y["compound"]

    skf = StratifiedKFold(n_splits=args.splits)
    pred_score_control = np.array([])
    pred_score_treatment = np.array([])
    mean_accuracy = []
    if type(X) == np.ndarray:
        for i, (train_index, test_index) in tqdm(
            enumerate(skf.split(X, y)), desc="K fold CV"
        ):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgs = model.fit(X_train, y_train)
            pred_score_control = np.append(
                pred_score_control, lgs.predict_proba(X_test[y_test == "DMSO"])[:, 0]
            )
            pred_score_treatment = np.append(
                pred_score_treatment, lgs.predict_proba(X_test[y_test == "taxol"])[:, 1]
            )
            mean_accuracy.append(lgs.score(X_test, y_test))
    #         print(y_test, lgs.predict_proba(X_test[y_test==0])[:,0], lgs.predict_proba(X_test[y_test==1])[:,1])
    elif type(X) == pd.core.frame.DataFrame:
        for i, (train_index, test_index) in tqdm(
            enumerate(skf.split(X, y)), desc="K fold CV"
        ):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = (
                data_standardization(X.iloc[train_index]),
                data_standardization(X.iloc[test_index]),
            )
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            lgs = model.fit(X_train, y_train)
            pred_score_control = np.append(
                pred_score_control, lgs.predict_proba(X_test[y_test == "DMSO"])[:, 1]
            )
            pred_score_treatment = np.append(
                pred_score_treatment, lgs.predict_proba(X_test[y_test == "taxol"])[:, 1]
            )
            mean_accuracy.append(lgs.score(X_test, y_test))
    return {
        perc: [
            np.mean(mean_accuracy),
            np.std(mean_accuracy),
            np.mean(pred_score_control),
            np.std(pred_score_control),
            np.mean(pred_score_treatment),
            np.std(pred_score_treatment),
        ]
    }


def multi_kfold_train_bag(perc, args, data, model, control=0, treatment=1, verbose=0):
    X, y = data_label_split(data)
    y = y["compound"]
    skf = StratifiedKFold(n_splits=args.splits)
    pred_score_control = np.array([])
    pred_score_treatment = np.array([])
    mean_accuracy = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('start perc %s, split %s' % (perc,i))
        X_train, X_test = (
            data_standardization(X.iloc[train_index]),
            data_standardization(X.iloc[test_index]),
        )
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = pd.concat([X_train, y_train], axis=1, sort=False)
        X_train, y_train = data_label_split(generate_data_set(args.bagsize, perc, X_train, treatment, control))
        y_train = y_train['compound']
        X_test = pd.concat([X_test, y_test], axis=1, sort=False)
        valida_dataset = dmso_taxol_ProfileBag(
            X_test,
            int(args.num_bags_train / args.splits),
            args.mean_bag_length,
            args.var_bag_length,
            perc,
            treatment,
            control,
            args.batch_size,
            0.5,
        )
        valida_loader = D.DataLoader(valida_dataset, batch_size=1, shuffle=True)
        lgs = model.fit(X_train, y_train)
        acc_control, acc_treat, pred_score_cont, pred_score_treat = test_bag_model(
                    lgs, valida_loader)
        pred_score_control = np.append(pred_score_control, pred_score_cont)
        pred_score_treatment = np.append(pred_score_treatment, pred_score_treat)
            
        mean_accuracy.append(np.mean(acc_control+acc_treat))
    if args.save_score:
        with open('%s_%f.txt' % (model, perc),'w') as f:
            f.write(','.join(["%.4f"%i for i in pred_score_control.tolist()])+'\n')
            f.write(','.join(["%.4f"%i for i in pred_score_treatment.tolist()]))
            
    return {
        perc: [
            np.mean(mean_accuracy),
            np.std(mean_accuracy),
            np.mean(pred_score_control),
            np.std(pred_score_control),
            np.mean(pred_score_treatment),
            np.std(pred_score_treatment),
        ]
    }

        
def test_bag_model(model, loader):
    pred_score_control = []
    pred_score_treat = []
    acc_control = []
    acc_treat = []
    
    # data will be bags with number of batch_size
    for batch_idx, (data, bag_labels) in enumerate(loader):
        data, bag_labels = data.squeeze(0).float(), bag_labels.squeeze(0).view(-1)
        # bag_data here is one bag only
        for bag_data, bag_label in zip(data,bag_labels):
            prob = model.predict_proba(bag_data.numpy())[:,0].mean()
            if int(bag_label.item()) == 0:
                pred_score_control.append(prob)
                acc_control.append(prob>=0.5)
            else:
                pred_score_treat.append(prob)
                acc_treat.append(prob<0.5)
    return acc_control, acc_treat, pred_score_control, pred_score_treat
        
        
def train(args, data, model, verbose, parallel=True, bag_perc=0.5):
    if parallel:
        results = multi_mini_noise_signal_cv(
            args, data, "taxol", "DMSO", model, verbose, bag_perc
        )
    else:
        results = mini_noise_signal_cv(
            args.bagsize, data, "taxol", "DMSO", model, verbose, bag_perc
        )

    results = pd.DataFrame.from_dict(results, orient="index")

    results.columns = [
        "mean_accuracy",
        "std_accuracy",
        "mean_pred_score_control",
        "std_pred_score_control",
        "mean_pred_score_treatment",
        "std_pred_score_treatment",
    ]

    model_name = str(model).split("(")[0]
    feature_size = len(data_label_split(data)[0].columns)
    result_path = os.path.join(args.output_dir,"%s_sample%s_feature%s.csv" % (model_name, args.bagsize, feature_size))
    if os.path.exists(result_path):
        results.to_csv(
            result_path,
            mode="a",
            header=False,
        )
    else:
        results.to_csv(result_path)



def main():
    args = set_args()
    time = datetime.datetime.now()
    args.output_dir += time.strftime("%b%d/") 
    os.makedirs(args.output_dir,exist_ok=True)
    drop_NA_data=pd.read_csv(args.data_path, index_col=0)
    X, y = data_label_split(drop_NA_data)
    if args.use_nuclei:
        X = use_nuclei_feature(X)
    elif args.use_nuclei_gran:
        X = use_nuclei_gran_feature(X)
    X['Metadata_PlateID_Nuclei'] = drop_NA_data['Metadata_PlateID_Nuclei'].tolist()
    X = normalize_by_group(X,'Metadata_PlateID_Nuclei')
    X.dropna('columns',inplace=True)
    X['compound'] = drop_NA_data['compound'].tolist()

    models = [KNeighborsClassifier(30),
              LogisticRegression(max_iter = 1000, solver = "saga", n_jobs = -1),
              RandomForestClassifier(min_samples_split=50, random_state=0),
              MLPClassifier(solver="adam", max_iter=100)]
    
    envs = os.environ
    if "SLURM_ARRAY_TASK_ID" in envs:
        model = models[int(envs['SLURM_ARRAY_TASK_ID'])]
    else:
        model = models[1]
    print('using model %s, data %s' % (str(model).split("(")[0], args.data_path))
    train(args, X, model, 0)
    
    
if __name__ == "__main__":
    main()
