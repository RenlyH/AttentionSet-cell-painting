def data_label_split(data):
    meta_list = []
    for i in data.columns:
        if "Meta" in i:
            meta_list.append(i)
    label = data[["compound"]]
    train = data.drop(meta_list + ["compound", "concentration", "moa", "row ID"], axis = 1)
    return train, label
    
def generate_data_set(size, perc, data, treatment, control):
    treatment_data = data[data["compound"] == treatment]
    control_data = data[data["compound"] == control]
    num_treat = int(size * perc)
    return treatment_data.sample(num_treat).append(control_data.sample(size - num_treat)).sample(frac = 1).reset_index(drop=True)