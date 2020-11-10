import pandas as pd
import os

from data_loader import data_label_split
from data_loader import generate_data_set
from data_loader import dmso_taxol_ProfileBag
from data_loader import data_normalization

from deepset_exp import train
from deepset_exp import test
from deepset_exp import mini_noise_signal_cv

from model import SmallDeepSet
from model import FullDeepSet
from model import profile_AttSet

import argparse
import torch
import torch.utils.data as D 
import torch.nn as nn
import torch.optim as optim

torch.cuda.set_device(1)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch cell mixture bags Example')
parser.add_argument('--start', type=int, default=5, metavar='s',
                    help='start percentage (default: 5)')
parser.add_argument('--end', type=int, default=96, metavar='n',
                    help='end percentage (default: 96)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--splits', type=int, default=10, metavar='k',
                    help='Total splits for K-fold cross validation')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--mean_bag_length', type=int, default=100, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=10, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=100, metavar='NTrain',
                    help='number of bags in training set')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pool', type=str, default='mean', help='Pooling methods: mean, max, sum, min')
parser.add_argument('--thres', type=float, default=0.5, help='Activation threshold')
# parser.add_argument('--load', type=str, default=True, help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON! with GPU %s'%torch.cuda.current_device())

print('Loading data')

file_name = "week1_full_data.csv"#"mini_moa_data_drop_NA.csv"

drop_NA_data = pd.read_csv(file_name, index_col=0)
# sf_drop_NA_data = drop_NA_data[["compound", "concentration",
#                                 "moa", "row ID", "Iteration (#2)", "COND",
#                                "AreaShape_Area_Nuclei", "AreaShape_Compactness_Nuclei"]]
data = data_normalization(drop_NA_data)

# model = FullDeepSet(args.pool, args.thres)

# if args.cuda:
#     model.cuda()
    
# print(args)

# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


    
for i in range(args.start, args.end, 5):
    # define model
    model = profile_AttSet(481,args.thres)#FullDeepSet(args.pool, args.thres)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    results = mini_noise_signal_cv(i , i + 1, data, args.num_bags_train, args.mean_bag_length, args.var_bag_length, "taxol", "DMSO", model, optimizer, args.splits, args.epochs)
    feature_size = len(data_label_split(data)[0].columns)

    results = pd.DataFrame.from_dict(results, orient = 'index')

    results.columns = ["mean_accuracy", "std_accuracy",
                "mean_control_accuracy", "std_control_accuracy", 
                "mean_treat_accuracy", "std_treat_accuracy", 
                "mean_pred_score_control", "std_pred_score_control",
                "mean_pred_score_treatment", "std_pred_score_treatment"]

    
    if os.path.exists("attset_%.1f_bags%d_bagsize%d_feature%d.csv"%(args.thres, args.num_bags_train, args.mean_bag_length, feature_size)):
        results.to_csv("attset_%.1f_bags%d_bagsize%d_feature%d.csv"%(args.thres, args.num_bags_train, args.mean_bag_length, feature_size), mode='a', header=False)
    else:
        results.to_csv("attset_%.1f_bags%d_bagsize%d_feature%d.csv"%(args.thres, args.num_bags_train, args.mean_bag_length, feature_size))
        

#     if os.path.exists("deepset_%s%.1f_bags%d_bagsize%d_feature%d.csv"%(args.pool, args.thres, args.num_bags_train, args.mean_bag_length, feature_size)):
#         results.to_csv("deepset_%s%.1f_bags%d_bagsize%d_feature%d.csv"%(args.pool, args.thres, args.num_bags_train, args.mean_bag_length, feature_size), mode='a', header=False)
#     else:
#         results.to_csv("deepset_%s%.1f_bags%d_bagsize%d_feature%d.csv"%(args.pool, args.thres, args.num_bags_train, args.mean_bag_length, feature_size))