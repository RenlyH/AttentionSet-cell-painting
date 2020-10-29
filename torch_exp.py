import pandas as pd 
import numpy as np
from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold


import torch.nn as nn
import torch.utils.data as D 
from torch.autograd import Variable


from data_loader import data_label_split
from data_loader import generate_data_set
from data_loader import dmso_taxol_ProfileBag

import multiprocessing as mp

def train(epoch, loader, model, opt):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in tqdm(enumerate(loader), desc = 'For %d epoch' % epoch): 
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
#     print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.item(), train_error))
    return(train_loss, train_error)
    
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
    return acc_control, acc_treat, pred_score_control, pred_score_treat

def mini_noise_signal_cv(data, num_bag, bag_size_mean, bag_size_std, treatment, control, model, optimizer, splits, epochs):
    mean_control_accuracy=[]
    std_control_accuracy=[]
    mean_treat_accuracy=[]
    std_treat_accuracy=[]
    mean_pred_score_control = []
    std_pred_score_control = []
    mean_pred_score_treatment = []
    std_pred_score_treatment = []

    
    # Set different percentage of treatment v.s. control 
    for j in tqdm(range(5,96,5), desc = "training at different percent"):
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
            train_dataset = dmso_taxol_ProfileBag(X_train, num_bag, bag_size_mean, bag_size_std, j/100, treatment, control, 0.5)
            valida_dataset = dmso_taxol_ProfileBag(X_test, num_bag, bag_size_mean, bag_size_std, j/100, treatment, control, 0.5)
            train_loader = D.DataLoader(train_dataset, batch_size=1, shuffle=True)
            valida_loader = D.DataLoader(valida_dataset, batch_size=1, shuffle=True)
            
            train_loss, train_error = train(epochs, train_loader, model, optimizer)
            acc_control, acc_treat, pred_score_control, pred_score_treat = test(model, valida_loader)
            
            acc_control_list+=acc_control
            acc_treat_list+=acc_treat
            pred_score_control_list+=pred_score_control
            pred_score_treat_list+=pred_score_treat
            
        mean_control_accuracy.append(np.mean(acc_control_list))
        std_control_accuracy.append(np.std(acc_control_list))
        mean_treat_accuracy.append(np.mean(acc_treat_list))
        std_treat_accuracy.append(np.std(acc_treat_list))
        
        mean_pred_score_control.append(np.mean(pred_score_control_list))
        std_pred_score_control.append(np.std(pred_score_control_list))
        mean_pred_score_treatment.append(np.mean(pred_score_treat_list))
        std_pred_score_treatment.append(np.std(pred_score_treat_list))
    return (mean_control_accuracy, std_control_accuracy, 
            mean_treat_accuracy, std_treat_accuracy, 
            mean_pred_score_control, std_pred_score_control,
            mean_pred_score_treatment, std_pred_score_treatment)


