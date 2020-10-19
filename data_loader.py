import pandas

'''
Given pandas.DataFrame type data with rows as individual cells and columns as features.
data must contain "compound", "concentration", "moa", "row ID"
Exclude Meta features and split X, y and return them.
'''
def data_label_split(data:pandas.DataFrame)->pandas.DataFrame:
    meta_list = []
    for i in data.columns:
        if "Meta" in i:
            meta_list.append(i)
    label = data[["compound"]]
    train = data.drop(meta_list + ["compound", "concentration", "moa", "row ID", "Iteration (#2)", "COND"], axis = 1)
    return train, label
    
'''
Generate sample data with specifed percent of PC & NC
'''
def generate_data_set(size:int, perc:float, data:pandas.DataFrame, treatment:str, control:str)->pandas.DataFrame:
    treatment_data = data[data["compound"] == treatment]
    control_data = data[data["compound"] == control]
    num_treat = int(size * perc)
    return treatment_data.sample(num_treat).append(control_data.sample(size - num_treat)).sample(frac = 1).reset_index(drop=True)