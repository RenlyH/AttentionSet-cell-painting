from data_loader import generate_data_set, data_label_split

def mini_noise_signal_cv(size, data, treatment, control, model, cv, verbose):
    mean_mean_accuracy=[]
    std_mean_accuracy=[]
    mean_z_score_control=[]
    std_z_score_control=[]
    mean_z_score_treatment=[]
    std_z_score_treatment=[]
    for i in range(5,96,5):
        mini_batch = generate_data_set(size, i/100, data, treatment, control)
        X, y = data_label_split(mini_batch)
#         y = y["compound"]
#         lb = LabelEncoder()
#         y_true = lb.fit_transform(y['compound']) # 0 for DMSO, 1 for taxol
        mean_accuracy, z_score_control, z_score_treatment = z_score(cv, X, y, model,"DMSO", "taxol", verbose = verbose)
        mean_mean_accuracy.append(np.mean(mean_accuracy))
        std_mean_accuracy.append(np.std(mean_accuracy))
        mean_z_score_control.append(np.mean(z_score_control))
        std_z_score_control.append(np.std(z_score_control))
        mean_z_score_treatment.append(np.mean(z_score_treatment))
        std_z_score_treatment.append(np.std(z_score_treatment))
    return mean_mean_accuracy, std_mean_accuracy, mean_z_score_control, std_z_score_control, mean_z_score_treatment, std_z_score_treatment



def z_score(splits, X, y, model, control = 0, treatment = 1, verbose = 0):
    skf = StratifiedKFold(n_splits = splits)
    z_score_control = []
    z_score_treatment = []
    mean_accuracy = []
    if type(X) == np.ndarray:
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index , "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgs = model.fit(X_train, y_train)
            z_score_control.append(lgs.predict_proba(X_test[y_test==control])[:,0].mean())
            z_score_treatment.append(lgs.predict_proba(X_test[y_test==treatment])[:,1].mean())
            mean_accuracy.append(lgs.score(X_test, y_test))
#         print(y_test, lgs.predict_proba(X_test[y_test==0])[:,0], lgs.predict_proba(X_test[y_test==1])[:,1])
    elif type(X) == pd.core.frame.DataFrame:
         for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if verbose != 0:
                print("Fold %d" % i, "TRAIN:", train_index , "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lgs = model.fit(X_train, y_train)
            z_score_control.append(lgs.predict_proba(X_test[y_test==control])[:,0].mean())
            z_score_treatment.append(lgs.predict_proba(X_test[y_test==treatment])[:,1].mean())
            mean_accuracy.append(lgs.score(X_test, y_test))
    return mean_accuracy, z_score_control, z_score_treatment