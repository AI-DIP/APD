#%%
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from src.adversarial_prototype_decomposition.classifier import classifiers as  apd
from imblearn.under_sampling import ClusterCentroids
from sklearn.preprocessing import LabelEncoder

#%% SET_PARAMS
N_PROTO = 3
MAX_DEPTH = 3
#%% LOAD DATASET
path_train_data = "Data\\banana.csv"
train_data = pd.read_csv(path_train_data, sep=",")
ohe = LabelEncoder()

cols = [col for col in train_data.columns if col not in ["Class"]]
X_train = train_data.loc[:, cols].values
feature_names = cols
y_train = train_data.loc[:, "Class"].values
y_train = ohe.fit_transform(y_train)

#%% CREATE MODEL

base_estimator = DecisionTreeClassifier(max_depth=MAX_DEPTH)
estimator = apd.APD_ClassifierScaler(
    type="apd2",
    base_estimator= base_estimator,
    min_support=100,
    unbalanced_rate= 0.1,
    proto_selection= ClusterCentroids(sampling_strategy={0: N_PROTO, 1: N_PROTO}),
)
# %% FIT MODEL
estimator.fit(X_train, y_train)

# %% predict proba
proba = estimator.predict_proba(X_train)
dist,regions = estimator.get_competency(X_train)
test = estimator.proto_ensemble_.assign_regions(X_train, estimator.regions_)


# %%
base_estimator.fit(X_train,y_train)

# %%
estimator.regions_
# %%
