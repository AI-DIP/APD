# %% Test for a single model APD or DecisionTree. For APD it also plots prototypes and regions using UMAP
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import tempfile
import os
import numpy as np
from apdlib import classifiers as apd

from apdlib.apd import APD2, APDBase
from apdlib.sampler.glvq_sampler import GLVQ_Sampler
from utils.plot_utils import get_plot_regions_centres, get_plot, get_prototypes_plot_MDS
from utils.mlflow_utils import save_fig_as_artefact, save_pandas_as_artefact
from utils.apd_utils import get_proto_info, get_region_info
#%%
TEST_RUN = False
APD_RUN = True
MDS_RUN = False
DRAW_DT_PLOT = False

CCP = [0]
N_PROTO = [10]
DT_MAX_DEPTH = [4]

#%%
# 1. Zaladuj dane
path_train_data = "Y:/DDabrowski/Dataset/Prepared_UT1_v5_048.csv"
train_data = pd.read_csv(path_train_data, sep=",")
ohe = LabelEncoder()
cols = [col for col in train_data.columns if col not in ["LABEL", "id", "id.1", "id.2", "id.3", "Applied torque"]]
X_train = train_data.loc[:, cols].values
feature_names = cols
y_train = train_data.loc[:, "LABEL"].values
y_train = ohe.fit_transform(y_train)
class_names = ["0", "1"]

path_test_data1 = "Y:/DDabrowski/Dataset/Prepared_UT2_v5.csv"
test1_data = pd.read_csv(path_test_data1, sep=",")
path_test_data2 = "Y:/DDabrowski/Dataset/Prepared_UT3_v5.csv"
test2_data = pd.read_csv(path_test_data2, sep=",")
test_data = pd.concat([test1_data, test2_data])
X_test = test_data.loc[:, cols].values
y_test = test_data.loc[:, "LABEL"].values
y_test = ohe.transform(y_test)

ccp = CCP[0]
nproto2 = N_PROTO[0]
max_depth = DT_MAX_DEPTH[0]
n_proto = nproto2

# Chwilowy katalog
tmp_dir = tempfile.mkdtemp()

#Models configuration
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, ccp_alpha=ccp)
estimator = apd.APD_ClassifierScaler(
    type="apd2",
    base_estimator=clf,
    min_support=200,
    unbalanced_rate=0.1,
    proto_selection=GLVQ_Sampler(prototype_n_per_class=np.array([n_proto, n_proto]),
                                 solver_params={"step_size": 0.1,
                                                "max_runs": 200,
                                                "batch_size": 128, })
    # ClusterCentroids(sampling_strategy={0: n_proto, 1: n_proto}),
)
estimator = clf #Note here

# Trening
# %%
estimator.fit(X_train, y_train)
# Predykcja - na zbiorze testowym
y_pred = estimator.predict(X_test)

# Metryki
bacc = balanced_accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Test set")
print({"ACC": acc, "BACC": bacc, "F1": f1})

#Predykcja na zbiorze treningowym zCV
res = cross_validate(estimator, X_train, y_train, cv=5, scoring="accuracy", n_jobs=5)
me = np.mean(res["test_score"])
print(f"CV={me}")
#%% Wypisanie złożoności poszczególnych modeli
print("===================== TREES ====================")
if type(estimator) == DecisionTreeClassifier:
    dt = export_text(estimator, feature_names=feature_names, decimals=4)
else:
    for reg,tree in estimator.fitted_base_models_.items():
        dt = export_text(tree,feature_names=feature_names, decimals=4)
    print(f"Regions = {len(estimator.regions_)}")
print(dt)

#%% Wyswietlenie prototypów na ekranie z wykorzystaniem UMAP
proto = estimator.proto_ensemble_.proto
proto_labels = estimator.proto_ensemble_.proto_labels
regs = estimator.regions_
mes = np.zeros((len(regs), proto.shape[1]))
dfs = np.zeros((len(regs), proto.shape[1]))
from visualization import umap_classification_pipeline

X_umap, scaler_umap, umap_umap = umap_classification_pipeline(X_train,y_train,["normal","f1"], return_umap=True)

for i,reg in enumerate(regs):
    a,b = APDBase.unpairCantor(reg)
    proto_umap = umap_umap.transform(scaler_umap.transform(proto[[a,b],:]))
    plt.plot(proto_umap[:,0], proto_umap[:,1], c='r', alpha=0.3)
    mes[i,:] = 0.5*(proto[a,:] + proto[b,:])
    dfs[i,:] = np.abs(proto[a,:] - proto[b,:])
plt.show()

