# %%
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import tempfile
import os
import numpy as np
from src.adversarial_prototype_decomposition.classifier import classifiers as apd

from src.adversarial_prototype_decomposition.base.apd import APD2, APDBase
from src.adversarial_prototype_decomposition.sampler.glvq_sampler import GLVQ_Sampler
from utils.plot_utils import get_plot_regions_centres, get_plot, get_prototypes_plot_MDS
from utils.mlflow_utils import save_fig_as_artefact, save_pandas_as_artefact
from src.adversarial_prototype_decomposition.utils.apd_utils import get_proto_info, get_region_info
from joblib import Parallel, delayed

def asses_apd(X_train,y_train,X_test,y_test,max_depth,ccp,n_proto,ur):
    print(f"max_depth = {max_depth}, ccp = {ccp}")
    # Trening
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, ccp_alpha=ccp)
    estimator = apd.APD_ClassifierScaler(
        type="apd2",
        base_estimator=clf,
        min_support=200,
        unbalanced_rate=ur,
        proto_selection=GLVQ_Sampler(prototype_n_per_class=np.array([n_proto, n_proto]),
                                     solver_params={"step_size": 0.1,
                                                    "max_runs": 200,
                                                    "batch_size": 128, }))

    estimator.fit(X_train, y_train)
    # Predykcja
    y_pred = estimator.predict(X_test)

    # Metryki
    bacc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Test set")
    print({"ACC": acc, "BACC": bacc, "F1": f1})

    res = cross_validate(estimator, X_train, y_train, cv=5, scoring="accuracy", n_jobs=5)
    me = np.mean(res["test_score"])
    print(f"CV={me}")
    print("===================== TREES ====================")

    print(f"Regions = {len(estimator.regions_)}")
    # print(dt)
    res_all = {"CV":me,
                "estimator":estimator,
                "max_depth":max_depth,
                "ccp":ccp,
                "acc":acc,
                "f1":f1,
                "balance_rate":ur,
                "proto_n":n_proto,
                "regions":len(estimator.regions_)}
    return res_all


#%%
TEST_RUN = False
APD_RUN = True
MDS_RUN = False
DRAW_DT_PLOT = False

CCPS = [0,0.001,0.005,0.01,0.05,0.1]
N_PROTOS = [2,3,4,5,6,7,8]
DT_MAX_DEPTHS = [3,4,5,6,7,8,9]
URS = [0.1,0.2,0.3]

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

# Chwilowy katalog
tmp_dir = tempfile.mkdtemp()

# %%
res_all = Parallel(n_jobs=55, verbose=1)(
    delayed(asses_apd)(X_train,y_train,X_test,y_test,max_depth,ccp,n_proto,ur)
    for n_proto in N_PROTOS
    for ur in URS
    for max_depth in DT_MAX_DEPTHS
    for ccp in CCPS
)

res_df = pd.DataFrame(res_all)
res_df.to_pickle("Results/res_all.pickle")
res_df.to_csv("Results/res_all.csv")
id = res_df["CV"].argmax()
estimator = res_df.loc[id, "estimator"]
#%%
proto = estimator.proto_ensemble_.proto
proto_labels = estimator.proto_ensemble_.proto_labels
regs = estimator.regions_
mes = np.zeros((len(regs), proto.shape[1]))
dfs = np.zeros((len(regs), proto.shape[1]))
from src.adversarial_prototype_decomposition.utils.visualization import umap_classification_pipeline

X_umap, scaler_umap, umap_umap = umap_classification_pipeline(X_train,y_train,["normal","f1"], return_umap=True)

for i,reg in enumerate(regs):
    a,b = APDBase.unpairCantor(reg)
    proto_umap = umap_umap.transform(scaler_umap.transform(proto[[a,b],:]))
    plt.plot(proto_umap[:,0], proto_umap[:,1], c='r', alpha=0.3)
    mes[i,:] = 0.5*(proto[a,:] + proto[b,:])
    dfs[i,:] = np.abs(proto[a,:] - proto[b,:])
plt.show()