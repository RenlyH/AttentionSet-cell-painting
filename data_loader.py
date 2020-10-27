import pandas
import torch
import torch.utils.data as D 
import numpy as np

'''
Given pandas.DataFrame type data with rows as individual cells and columns as features.
data must contain "compound", "concentration", "moa", "row ID"
Exclude Meta features and split X, y and return them.
'''
def data_label_split(data:pandas.DataFrame) -> pandas.DataFrame:
    drop_list = []
    for i in data.columns:
        if "Meta" in i or i in ["compound", "concentration", "moa", "row ID", "Iteration (#2)", "COND"]:
            drop_list.append(i)
    label = data[["compound"]]
    train = data.drop(drop_list, axis = 1)
    return train, label
    
'''
Generate sample data with specifed percent of PC & NC
'''
def generate_data_set(size:int, perc:float, data:pandas.DataFrame, treatment:str, control:str) -> pandas.DataFrame:
    treatment_data = data[data["compound"] == treatment]
    control_data = data[data["compound"] == control]
    num_treat = int(size * perc)
    return treatment_data.sample(num_treat).append(control_data.sample(size - num_treat)).sample(frac = 1).reset_index(drop=True)

# takes the df with "compound" columns
class dmso_taxol_ProfileBag(D.Dataset):
    def __init__(self, df:pandas.DataFrame, size:int, bag_mean_size, bag_std_size, perc:float, treatment:str, control:str, merged_perc:float):
        self.df = df
        self.size = size
        merged_perc = 0.5
        self.merged_pos_size = int(size * merged_perc)
        self.unmerged_neg_size = int(size * (1 - merged_perc))
        
        self.bag_mean_size = bag_mean_size
        self.bag_std_size = bag_std_size
        self.perc = perc
        self.treatment = treatment
        self.control = control
        self.bag_size = np.random.normal(bag_mean_size, bag_std_size, size).astype(int)


        self.treatment_data = self.df[self.df["compound"] == self.treatment]
        self.control_data = self.df[self.df["compound"] == self.control]
        self.treatment_size = (self.bag_size * self.perc).astype(int)
        self.control_size = self.bag_size - self.treatment_size
        
    def _get_sampling_index(self):
        treat_index_list = []
        control_index_list = []

        treat_length = len(self.treatment_data)
        control_length = len(self.control_data)
        
        index = np.arange(treat_length)
        for i in range(self.merged_pos_size):
            if len(index) < self.treatment_size[i]:
                index = np.arange(treat_length)
            z = np.random.choice(index, self.treatment_size[i], replace=False)
            index = index[~np.isin(index, z)]
            treat_index_list.append(z)

        index = np.arange(control_length)
        for i in range(self.merged_pos_size):
            if len(index) < self.control_size[i]:
                index = np.arange(control_length)
            z = np.random.choice(index, self.control_size[i], replace=False)
            index = index[~np.isin(index, z)]
            control_index_list.append(z)
        return treat_index_list, control_index_list
        
        
    def __getitem__(self, index):
        if index < self.merged_pos_size:
            treat_index_list, control_index_list= self._get_sampling_index()
            merged_data = self.treatment_data.iloc[treat_index_list[index]].append(self.control_data.iloc[control_index_list[index]]) 
            X, y = data_label_split(merged_data.sample(frac=1))
            return torch.from_numpy(X.values), [torch.tensor([1.0]), list(y["compound"])]
        else:
            unmerged_bag_size = np.random.normal(self.bag_mean_size, self.bag_std_size, 1).astype(int).item()
            X, y = data_label_split(self.control_data.sample(unmerged_bag_size))
            return torch.from_numpy(X.values), [torch.tensor([0.0]), list(y["compound"])]
        
    def __len__(self):
        return self.size
    
    
    
