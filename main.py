import pandas as pd

from data_loader import data_label_split
from data_loader import generate_data_set
from data_loader import dmso_taxol_ProfileBag

from torch_exp import train
from torch_exp import test
from torch_exp import mini_noise_signal_cv

import argparse
import torch
import torch.utils.data as D 
import torch.nn as nn
import torch.optim as optim

import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch cell mixture bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--splits', type=int, default=5, metavar='k',
                    help='Total splits for K-fold cross validation')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')

parser.add_argument('--mean_bag_length', type=int, default=20, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=1, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
# parser.add_argument('--load', type=str, default=True, help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load data')
drop_NA_data = pd.read_csv("mini_moa_data_drop_NA.csv",index_col=0)#pd.read_csv("moa_data_drop_NA.csv", index_col=0)


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=481, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        elif self.pool == "min":
            x = x.min(dim=1)[0]
        x = self.dec(x)
        return x, torch.ge(x, 0.5)

full_deepset = SmallDeepSet()

if args.cuda:
    full_deepset.cuda()
    
optimizer = optim.Adam(full_deepset.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
results = mini_noise_signal_cv(drop_NA_data, args.num_bags_train, args.mean_bag_length, args.var_bag_length, "taxol", "DMSO", full_deepset,optimizer, args.splits, args.epochs)

results = pd.DataFrame(list(results)).transpose()

results.columns = ["mean_control_accuracy", "std_control_accuracy", 
            "mean_treat_accuracy", "std_treat_accuracy", 
            "mean_pred_score_control", "std_pred_score_control",
            "mean_pred_score_treatment", "std_pred_score_treatment"]
results.to_csv("exp_bagsize%d_full_feature.csv"%(args.num_bags_train),index = False)