import os
import pandas as pd 
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from model import SmallDeepSet

import torch.nn as nn
import torch.utils.data as D 
from torch.autograd import Variable

from data_loader import data_label_split
from data_loader import generate_data_set
from data_loader import dmso_taxol_ProfileBag
import torch.nn.functional as F
import torch.optim as optim

def train(epoch, loader, model, opt, verbose_idx):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label[0]
        data, bag_label = data.float().cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        opt.zero_grad()
        # calculate loss and metrics
        y_prob, y_hat = model(data)
        loss = nn.BCELoss()(y_prob, bag_label)
        train_loss += loss.data.cpu()
        error = 1-y_hat.float().eq(bag_label).cpu().float().mean().item()
        train_error += error
        # backward pass
        loss.backward()
        # step
        opt.step()

    # calculate mean loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    if epoch%verbose_idx==0:
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.item(), train_error))
#     return(train_loss, train_error)
    
def test(model, loader):
    model.eval()
    pred_score_control = []
    pred_score_treat = []
    acc_control = []
    acc_treat = []
    
    for batch_idx, (data, label) in enumerate(loader): 
        bag_label = label[0]
        data, bag_label = data.float().cuda(), bag_label.cuda()
        y_prob, y_hat = model(data)

        if int(bag_label.item()) == 0:
            pred_score_control.append(y_prob.cpu().float().item())
            acc_control.append(y_hat.float().eq(bag_label).cpu().float().mean().item())
        else:
            pred_score_treat.append(y_prob.cpu().float().item())
            acc_treat.append(y_hat.float().eq(bag_label).cpu().float().mean().item())
    print("Control accuracy:%.3f, Treat accuracy:%.3f, overally accuray:%.3f" %(np.mean(acc_control),np.mean(acc_treat),np.mean(acc_control+acc_treat)))
    return acc_control, acc_treat, pred_score_control, pred_score_treat

def mini_noise_signal_cv(start, end, data, num_bag, bag_size_mean, bag_size_std, 
                         treatment, control, model, optimizer, splits, epochs):
    dic = {}
    # Set different percentage of treatment v.s. control 
    for j in tqdm(range(start, end, 5), desc = "training at different percent"):
        X, y = data_label_split(data)
        y = y["compound"]

        acc_control_list = []
        acc_treat_list = []
        pred_score_control_list = []
        pred_score_treat_list = []
        # Stratified K fold 
        skf = StratifiedKFold(n_splits = splits)
        for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), desc="%d fold cross validation"%splits):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
            X_train = pd.concat([X_train, y_train], axis=1, sort=False)
            X_test = pd.concat([X_test, y_test], axis=1, sort=False)

            # Redefine dataloader and train model at each fold
            train_dataset = dmso_taxol_ProfileBag(X_train, int(num_bag*(splits-1)/splits), bag_size_mean, bag_size_std, j/100, treatment, control, 0.5)
            valida_dataset = dmso_taxol_ProfileBag(X_test, int(num_bag/splits), bag_size_mean, bag_size_std, j/100, treatment, control, 0.5)
            train_loader = D.DataLoader(train_dataset, batch_size=1, shuffle=True)
            valida_loader = D.DataLoader(valida_dataset, batch_size=1, shuffle=True)
            # Start training 
            for epoch in range(epochs):
                train(epoch, train_loader, model, optimizer, 10)
            # Conduct testing
            print("At %dth fold testing: "%i, end="")
            acc_control, acc_treat, pred_score_control, pred_score_treat = test(model, valida_loader)
            
            acc_control_list+=acc_control
            acc_treat_list+=acc_treat
            pred_score_control_list+=pred_score_control
            pred_score_treat_list+=pred_score_treat
        dic[j/100] = [np.mean(acc_control_list+acc_treat_list), np.std(acc_control_list+acc_treat_list),
                          np.mean(acc_control_list), np.std(acc_control_list), 
                          np.mean(acc_treat_list), np.std(acc_treat_list),
                          np.mean(pred_score_control_list),np.std(pred_score_control_list),
                          np.mean(pred_score_treat_list), np.std(pred_score_treat_list)]
        return dic
# (mean_control_accuracy, std_control_accuracy, 
#             mean_treat_accuracy, std_treat_accuracy, 
#             mean_pred_score_control, std_pred_score_control,
#             mean_pred_score_treatment, std_pred_score_treatment)


if __name__=='__main__':
    drop_NA_data=pd.read_csv("mini_moa_data_drop_NA.csv", index_col=0)
    print("data loaded")
    sf_drop_NA_data = drop_NA_data[["compound", "concentration",
                                "moa", "row ID", "Iteration (#2)", "COND",
                               "AreaShape_Area_Nuclei", "AreaShape_Compactness_Nuclei"]]
    sf_deepset = SmallDeepSet().cuda()
    optimizer = optim.Adam(sf_deepset.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5)
    results = mini_noise_signal_cv(15, 21, sf_drop_NA_data, 10, 20, 1, "taxol", "DMSO", sf_deepset, optimizer, 3, 1)
    results = pd.DataFrame.from_dict(results, orient = 'index')

    results.columns = ["mean_total_accuracy", "std_total_accuracy",
                "mean_control_accuracy", "std_control_accuracy", 
                "mean_treat_accuracy", "std_treat_accuracy", 
                "mean_pred_score_control", "std_pred_score_control",
                "mean_pred_score_treatment", "std_pred_score_treatment"]
    if os.path.exists("exp_bagsize%d_full_feature.csv"%(300)):
        results.to_csv("exp_bagsize%d_full_feature.csv"%(300), mode='a', header=False)
    else:
        results.to_csv("exp_bagsize%d_full_feature.csv"%(300))
    

