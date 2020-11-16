import pandas as pd
import os

from data_loader import data_label_split
from data_loader import generate_data_set
from data_loader import dmso_taxol_ProfileBag
# from data_loader import data_normalization

from set_experiment import train
from set_experiment import test
from set_experiment import mini_noise_signal_cv

from model import SmallDeepSet
from model import FullDeepSet
from model import profile_AttSet

import argparse
import torch
import torch.utils.data as D 
import torch.nn as nn
import torch.optim as optim

torch.cuda.set_device(2)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch cell mixture bags Example')
parser.add_argument('--start', type=int, default=5, metavar='s',
                    help='start percentage (default: 5)')
parser.add_argument('--end', type=int, default=96, metavar='n',
                    help='end percentage (default: 96)')
parser.add_argument('--epochs', type=int, default=35, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--splits', type=int, default=5, metavar='k',
                    help='Total splits for K-fold cross validation')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--mean_bag_length', type=int, default=150, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=10, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=60, metavar='NTrain',
                    help='number of batches in dataset')
parser.add_argument('--batch_size', type=int, default=20, help='Number of batches')


parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pool', type=str, default='sum', help='Pooling methods: mean, max, sum, min')
parser.add_argument('--thres', type=float, default=0.5, help='Activation threshold')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON! with GPU %s'%torch.cuda.current_device())



file_name = "moa_data_drop_NA.csv" # "week1_full_data.csv"
drop_NA_data = pd.read_csv(file_name, index_col=0)
print('Data loaded')
data = drop_NA_data


print(args)

    
for i in range(args.start, args.end, 5):
    # define model
    model = FullDeepSet(args.pool, args.thres)
#     model = profile_AttSet(481,args.thres)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    results = mini_noise_signal_cv(i , i + 1, data, args.num_bags_train, args.mean_bag_length, args.var_bag_length, "taxol", "DMSO", args.batch_size, model, optimizer, args.splits, args.epochs)
    feature_size = len(data_label_split(data)[0].columns)

    results = pd.DataFrame.from_dict(results, orient = 'index')

    results.columns = ["mean_accuracy", "std_accuracy",
                "mean_control_accuracy", "std_control_accuracy", 
                "mean_treat_accuracy", "std_treat_accuracy", 
                "mean_pred_score_control", "std_pred_score_control",
                "mean_pred_score_treatment", "std_pred_score_treatment"]

    
#     if os.path.exists("deepset_att%.1f_bags%d*%d_bagsize%d_feature%d.csv"%(args.thres, args.num_bags_train, args.batch_size, args.mean_bag_length, feature_size)):
#         results.to_csv("deepset_att%.1f_bags%d*%d_bagsize%d_feature%d.csv"%(args.thres, args.num_bags_train, args.batch_size, args.mean_bag_length, feature_size), mode='a', header=False)
#     else:
#         results.to_csv("deepset_att%.1f_bags%d*%d_bagsize%d_feature%d.csv"%(args.thres, args.num_bags_train, args.batch_size, args.mean_bag_length, feature_size))
        

    if os.path.exists("deepset_%s%.1f_bags%d*%d_bagsize%d_feature%d.csv"%(args.pool, args.thres, args.num_bags_train,args.batch_size, args.mean_bag_length, feature_size)):
        results.to_csv("deepset_%s%.1f_bags%d*%d_bagsize%d_feature%d.csv"%(args.pool, args.thres, args.num_bags_train,args.batch_size, args.mean_bag_length, feature_size), mode='a', header=False)
    else:
        results.to_csv("deepset_%s%.1f_bags%d*%d_bagsize%d_feature%d.csv"%(args.pool, args.thres, args.num_bags_train,args.batch_size, args.mean_bag_length, feature_size))
