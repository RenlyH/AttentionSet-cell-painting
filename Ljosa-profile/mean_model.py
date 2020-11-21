# Used to reproduce Ljosa's work with the data they provided.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

os.chdir("/home/user/michigan/data/Ljosa-BBBC021/database")


def get_MOA(comp):
    return comp_moa[comp_moa["compound"] == comp]["moa"].values[0]


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    plt.savefig("/home/user/michigan/att_pooling/Ljosa-mean/confu.png")


# features will not be used in profiling
exclude_features = [
    "Nuclei_Location_Center_X",
    "Nuclei_Location_Center_Y",
    "Cells_Location_Center_X",
    "Cells_Location_Center_Y",
    "Cytoplasm_Location_Center_X",
    "Cytoplasm_Location_Center_Y",
]

# Loading data
GroundTruth = pd.read_csv(
    "supplement_GroundTruth.txt", sep="\t", names=["compound", "concentration", "moa"]
)
comp_moa = GroundTruth.drop("concentration", axis=1).drop_duplicates()

namelist = []
with open("supplement_Image.sql") as f:
    for i in f.readlines():
        if "NULL" in i:
            namelist.append(i.split("`")[1])
Image = pd.read_csv("supplement_Image.txt", sep="\t", names=namelist)

namelist = []
with open("supplement_Object.sql") as f:
    for i in f.readlines():
        if "NULL" in i:
            namelist.append(i.split("`")[1])
Object = pd.read_csv("supplement_Object.txt", sep="\t", names=namelist)
print("loading data complete")

# Labelling data with component and concentration
data = Object.merge(
    Image[
        [
            "TableNumber",
            "ImageNumber",
            "Image_Metadata_Compound",
            "Image_Metadata_Concentration",
        ]
    ],
    on=["TableNumber", "ImageNumber"],
)
data = data.drop(exclude_features, axis=1)


# Profiling
treatment_profile = data.groupby(
    ["Image_Metadata_Compound", "Image_Metadata_Concentration"], as_index=False
).mean()
treatment_profile = treatment_profile[
    treatment_profile["Image_Metadata_Compound"] != "DMSO"
].drop(["TableNumber", "ImageNumber", "ObjectNumber"], axis=1)
X = np.array(treatment_profile.drop("Image_Metadata_Compound", axis=1))
y = np.array(treatment_profile["Image_Metadata_Compound"])
print("profiling complete")

# 1NN classification, leave same components out
result = pd.DataFrame()
for i in range(len(X)):
    masked_index = y == y[i]
    X_train = X[masked_index == False]
    y_train = y[masked_index == False]

    X_test = X[i : (i + 1)]
    y_test = y[i]

    neigh = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="cosine")
    neigh.fit(X_train, y_train)

    pred = neigh.predict(X_test)
    result = result.append(
        {"y": get_MOA(y_test), "y_pred": get_MOA(pred[0])}, ignore_index=True
    )

confu_mat = confusion_matrix(
    result["y"], result["y_pred"], labels=result["y"].value_counts().sort_index().index
)
plot_confusion_matrix(confu_mat, classes=result["y"].value_counts().sort_index().index)
