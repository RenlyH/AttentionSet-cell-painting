import os

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from tqdm import tqdm

from data_loader import (
    data_label_split,
    data_standardization,
    dmso_taxol_ProfileBag,
    generate_data_set,
)
from model import SmallDeepSet


def train(args, epoch, loader, model, opt, verbose_idx):
    model.train()
    train_loss = 0.0
    train_error = 0.0
    for batch_idx, (data, bag_label) in enumerate(loader):
        data, bag_label = data.squeeze(0).float(), bag_label.squeeze(0)
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        opt.zero_grad()
        # calculate loss and metrics
        y_prob, y_hat = model(data)
        loss = nn.BCELoss()(y_prob, bag_label)

        train_loss += loss.data.cpu()
        error = 1 - y_hat.float().eq(bag_label).cpu().float().mean().item()
        train_error += error
        # backward pass
        loss.backward()
        # step
        opt.step()

    # calculate mean loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    if epoch % verbose_idx == 0:
        print(
            "Epoch: {}, Loss: {:.4f}, Train error: {:.4f}".format(
                epoch, train_loss.item(), train_error
            )
        )
    return (train_loss.item(), train_error)


def test(args, model, loader):
    model.eval()
    pred_score_control = []
    pred_score_treat = []
    acc_control = []
    acc_treat = []

    for batch_idx, (data, bag_label) in enumerate(loader):
        data, bag_label = data.squeeze(0).float(), bag_label.squeeze(0).view(-1)
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        y_prob, y_hat = model(data)

        for i in range(len(bag_label)):
            if int(bag_label[i].item()) == 0:
                pred_score_control.append(y_prob[i].cpu().float().item())
                acc_control.append(
                    y_hat[i].float().eq(bag_label[i]).cpu().float().mean().item()
                )
            else:
                pred_score_treat.append(y_prob[i].cpu().float().item())
                acc_treat.append(
                    y_hat[i].float().eq(bag_label[i]).cpu().float().mean().item()
                )
    print(
        "Control accuracy:%.3f, Treat accuracy:%.3f, overall accuracy:%.3f"
        % (np.mean(acc_control), np.mean(acc_treat), np.mean(acc_control + acc_treat))
    )
    return acc_control, acc_treat, pred_score_control, pred_score_treat


def mini_noise_signal_cv(
    start,
    end,
    data,
    treatment,
    control,
    model,
    args
):
    dic = {}
    # Set different percentage of treatment v.s. control
    for j in range(start, end, 5):
        X, y = data_label_split(data)
        y = y["compound"]

        acc_control_list = []
        acc_treat_list = []
        pred_score_control_list = []
        pred_score_treat_list = []
        # Stratified K fold
        skf = StratifiedKFold(n_splits=args.splits)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = (
                data_standardization(X.iloc[train_index]),
                data_standardization(X.iloc[test_index]),
            )
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train = pd.concat([X_train, y_train], axis=1, sort=False)
            X_test = pd.concat([X_test, y_test], axis=1, sort=False)

            # Redefine dataloader and train model at each fold
            train_dataset = dmso_taxol_ProfileBag(
                X_train,
                int(args.num_bags_train * (args.splits - 1) / args.splits),
                args.mean_bag_length,
                args.var_bag_length,
                j / 100,
                treatment,
                control,
                args.batch_size,
                0.5,
                True,
            )
            valida_dataset = dmso_taxol_ProfileBag(
                X_test,
                int(args.num_bags_train / args.splits),
                args.mean_bag_length,
                args.var_bag_length,
                j / 100,
                treatment,
                control,
                args.batch_size,
                0.5,
            )
            train_loader = D.DataLoader(train_dataset, batch_size=1, shuffle=True)
            valida_loader = D.DataLoader(valida_dataset, batch_size=1, shuffle=True)
            # Start training
            model.__init__(model.input_feature, model.pool, model.thres)
            if args.cuda:
                model.cuda()
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.reg,
            )

            minimum_error = float("inf")
            early_stop = []
            for epoch in range(args.epochs):
                epoch_result = []
                print("Train, Percent:%d, Fold: %d, " % (j, i), end="")
                train_loss, train_error = train(
                    args, epoch, train_loader, model, optimizer, 1
                )
#                 if train_loss >= 49:
#                     X_train.to_csv("bag_perc%d_fold%d.csv" % (j, i))
#                     break
                epoch_result.append(train_loss)
                epoch_result.append(train_error)
                # Conduct testing
                print("Test, Percent:%d, Fold:%d, " % (j, i), end="")
                acc_control, acc_treat, pred_score_control, pred_score_treat = test(
                    args, model, valida_loader
                )
                if 1 - np.mean(acc_control + acc_treat) < minimum_error:
                    minimum_error = 1 - np.mean(acc_control + acc_treat)
                    best_result = (
                        acc_control,
                        acc_treat,
                        pred_score_control,
                        pred_score_treat,
                    )

                epoch_result.append(1 - np.mean(acc_control))
                epoch_result.append(1 - np.mean(acc_treat))
                if len(early_stop) < 5:
                    early_stop.append(epoch_result)
                else:
                    early_stop.append(epoch_result)
                    early_stop.pop(0)
                # Stop if loss and training+testing error is close to 0 in 5 consecutive epochs
                if np.mean(early_stop) <= 1e-6:
                    break
            acc_control_list += best_result[0]
            acc_treat_list += best_result[1]
            pred_score_control_list += best_result[2]
            pred_score_treat_list += best_result[3]
            print(np.mean(best_result[0] + best_result[1]))
        dic[j / 100] = [
            np.mean(acc_control_list + acc_treat_list),
            np.std(acc_control_list + acc_treat_list),
            np.mean(acc_control_list),
            np.std(acc_control_list),
            np.mean(acc_treat_list),
            np.std(acc_treat_list),
            np.mean(pred_score_control_list),
            np.std(pred_score_control_list),
            np.mean(pred_score_treat_list),
            np.std(pred_score_treat_list),
        ]
        return dic


if __name__ == "__main__":
    drop_NA_data = pd.read_csv("mini_moa_data_drop_NA.csv", index_col=0)
    print("data loaded")
    sf_drop_NA_data = drop_NA_data[
        [
            "compound",
            "concentration",
            "moa",
            "row ID",
            "Iteration (#2)",
            "COND",
            "AreaShape_Area_Nuclei",
            "AreaShape_Compactness_Nuclei",
        ]
    ]
    sf_deepset = SmallDeepSet().cuda()
    optimizer = optim.Adam(
        sf_deepset.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5
    )
    results = mini_noise_signal_cv(
        15, 21, sf_drop_NA_data, 10, 20, 1, "taxol", "DMSO", sf_deepset, optimizer, 3, 1
    )
    results = pd.DataFrame.from_dict(results, orient="index")

    results.columns = [
        "mean_total_accuracy",
        "std_total_accuracy",
        "mean_control_accuracy",
        "std_control_accuracy",
        "mean_treat_accuracy",
        "std_treat_accuracy",
        "mean_pred_score_control",
        "std_pred_score_control",
        "mean_pred_score_treatment",
        "std_pred_score_treatment",
    ]
    if os.path.exists("exp_bagsize%d_full_feature.csv" % (300)):
        results.to_csv("exp_bagsize%d_full_feature.csv" % (300), mode="a", header=False)
    else:
        results.to_csv("exp_bagsize%d_full_feature.csv" % (300))
