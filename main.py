import argparse
import os
import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D

from data_loader import data_label_split, dmso_taxol_ProfileBag, generate_data_set, normalize_by_group, use_nuclei_feature, use_nuclei_gran_feature
from model import FullDeepSet, SmallDeepSet, profile_AttSet
from set_experiment import mini_noise_signal_cv, test, train


def initialize_model(model, opt):
    model.__init__(model.input_feature, model.pool, model.thres)
    opt = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg
    )
    return opt
# torch.cuda.set_device(2)


def set_args():
# Training settings
    parser = argparse.ArgumentParser(description="PyTorch cell mixture bags Example")
    parser.add_argument(
        "--start", type=int, default=5, metavar="s", help="start percentage (default: 5)"
    )
    parser.add_argument(
        "--end", type=int, default=96, metavar="n", help="end percentage (default: 96)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        metavar="k",
        help="Total splits for K-fold cross validation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--reg", type=float, default=10e-5, metavar="R", help="weight decay"
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
    parser.add_argument("--batch_size", type=int, default=20, help="Number of batches")


    parser.add_argument(
        "--seed", type=int, default=10, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--pool", type=str, default="sum", help="Pooling methods: mean, max, sum, min"
    )
    parser.add_argument("--use_nuclei", action="store_true", default=False, help="use nuclei features only")
    parser.add_argument("--use_nuclei_gran", action="store_true", default=False, help="use nuclei granularity features only")
    parser.add_argument("--thres", type=float, default=0.5, help="Activation threshold")
    parser.add_argument("--data_path", type=str, default='moa_data_drop_NA.csv', help="data path")
    parser.add_argument("--output_dir", type=str, default='result/')
    args = parser.parse_args()
    return args


def main():
    args = set_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("GPU is ON! with GPU %s" % torch.cuda.current_device())

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
    data = X
    feature_size = len(data_label_split(data)[0].columns)

    pools = ['att','mean','max','min']
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        args.pool = pools[int(os.environ['SLURM_ARRAY_TASK_ID'])]
        
    for i in range(args.start, args.end, 5):
        # define model
        if args.pool == 'att':
            model = profile_AttSet(feature_size, args.thres)
        else:
            model = FullDeepSet(feature_size, args.pool, args.thres)
        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg
        )

        results = mini_noise_signal_cv(
            i,
            i + 1,
            data,
            "taxol",
            "DMSO",
            model,
            args
        )

        results = pd.DataFrame.from_dict(results, orient="index")

        results.columns = [
            "mean_accuracy",
            "std_accuracy",
            "mean_control_accuracy",
            "std_control_accuracy",
            "mean_treat_accuracy",
            "std_treat_accuracy",
            "mean_pred_score_control",
            "std_pred_score_control",
            "mean_pred_score_treatment",
            "std_pred_score_treatment",
        ]

        res_path = os.path.join(args.output_dir, "%s_deepset_thres%.1f_bags%d*%d_bagsize%d_feature%d.csv"% (
                args.pool,
                args.thres,
                args.num_bags_train,
                args.batch_size,
                args.mean_bag_length,
                feature_size,
            ))
        
        if os.path.exists(res_path):
            results.to_csv(
                res_path,
                mode="a",
                header=False,
            )
        else:
            results.to_csv(res_path)

if __name__ == "__main__":
    main()
