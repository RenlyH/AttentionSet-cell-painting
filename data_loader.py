import pandas
import torch
import torch.utils.data as D 
import numpy as np
from sklearn.preprocessing import normalize

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
Generate sample data with specifed percent of PC & NC. "perc" is the percentage of positive instance in the postive bag. To train the model, we treat all instances within positive bag positive no matter where they originally from.
'''
def generate_data_set(size:int, perc:float, data:pandas.DataFrame, treatment:str, control:str, bag_perc:float=0.5) -> pandas.DataFrame:
    treatment_data = data[data["compound"] == treatment]
    control_data = data[data["compound"] == control]
    num_treat = int(size * perc * bag_perc)
    num_modified = int(size * (1-perc) * bag_perc)
    flag = treatment_data.shape[0]<num_treat
    treatment_data = treatment_data.sample(num_treat, replace = flag)
    flag = control_data.shape[0]<(size - num_treat)
    control_data = control_data.sample(size - num_treat,replace = flag).reset_index(drop = True)
    control_data.loc[0:num_modified-1,("compound")] = treatment
#     return treatment_data, control_data
    return treatment_data.append(control_data).sample(frac = 1).reset_index(drop=True)

'''
Sampling instance and form a bag from the the meta data (pandas dataframe)
'''
class dmso_taxol_ProfileBag(D.Dataset):
    def __init__(self, df:pandas.DataFrame, size:int, bag_mean_size, bag_std_size, perc:float, treatment:str, control:str, merged_perc:float=0.5):
        self.df = df
        self.size = size
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
        self.treatment_replace = len(self.treatment_data) < np.sum(self.treatment_size)
        self.control_replace = len(self.control_data) < np.sum(self.control_size)
        self.treat_index_list, self.control_index_list= self._get_sampling_index()
        
    def _get_sampling_index(self):
        treat_index_list = []
        control_index_list = []

        treat_length = len(self.treatment_data)
        control_length = len(self.control_data)
        
        index = np.arange(treat_length)
        for i in range(self.merged_pos_size):
            if len(index) < self.treatment_size[i]:
                index = np.arange(treat_length)
            z = np.random.choice(index, self.treatment_size[i], replace=self.treatment_replace)
            index = index[~np.isin(index, z)]
            treat_index_list.append(z)

        index = np.arange(control_length)
        for i in range(self.merged_pos_size):
            if len(index) < self.control_size[i]:
                index = np.arange(control_length)
            z = np.random.choice(index, self.control_size[i], replace=self.control_replace )
            index = index[~np.isin(index, z)]
            control_index_list.append(z)
        return treat_index_list, control_index_list
        
        
    def __getitem__(self, index):
        if index < self.merged_pos_size:
            merged_data = self.treatment_data.iloc[self.treat_index_list[index]].append(self.control_data.iloc[self.control_index_list[index]]) 
            X, y = data_label_split(merged_data.sample(frac=1))
            return torch.from_numpy(X.values), [torch.tensor([1.0]), list(y["compound"])]
        else:
            unmerged_bag_size = np.random.normal(self.bag_mean_size, self.bag_std_size, 1).astype(int).item()
            X, y = data_label_split(self.control_data.sample(unmerged_bag_size))
            return torch.from_numpy(X.values), [torch.tensor([0.0]), list(y["compound"])]
        
    def __len__(self):
        return self.size
    
    
'''
Normalize cell profiler extracted feature data enabling the convergence of ML model.
'''
def data_normalization(data:pandas.DataFrame)->pandas.DataFrame:
    X,y=data_label_split(data)
    return pandas.concat([pandas.DataFrame(normalize(X), columns = X.columns), y.reset_index(drop=True)], axis=1, sort = False )

