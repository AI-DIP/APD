# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""
#ToDo: https://github.com/omadson/fuzzy-c-means/blob/master/fcmeans/main.py
import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from apdlib.sampler.kmeans_sampler import SimpleClusterCentroids
from imblearn.under_sampling import ClusterCentroids
from sklearn.pipeline import Pipeline
from apdlib import classifiers as  apd
from sklearn.base import clone
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from mlflow_connector import MlFlowConnector
from GridSearch import GridSearch
from new_metric import mlflow_scoring_wrapper
from getParam import getCcpAlpha, getMaxDepth, getNPrototypes
import mlflow.data
import os

def login():
    if os.environ.get("MLFLOW_TRACKING_USERNAME",None) is None or os.environ.get("MLFLOW_TRACKING_PASSWORD",None) is None:
        user = input("USER: ")
        password = input("PASSWORD: ")
        os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()

login()

# folder = "Y:/Datasets/Datasets/KeelNorm"
# resFolder = "Data/Results/"
# files = ["banana","electricity-normalized"]
folder = "Y:/DDabrowski/Dataset"
resFolder = "Data/Results/"
files = ["Prepared_UT1_v5_048","Prepared_UT2_v5","Prepared_UT3_v5"]

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "passwor"

mlFlow = MlFlowConnector("http://192.168.10.40:5000")
mlFlow.start()
# estimator = DecisionTreeClassifier()
# param_grid = {
#     "max_depth":[3,4,5,6,7,10],
#     "ccp_alpha":[0,0.005,0.01,0.02,0.03]
# }
additional_params = None
param_grid = {
    "type":["apd2"],
    "base_estimator":[
        DecisionTreeClassifier(max_depth=3, ccp_alpha=0),
        DecisionTreeClassifier(max_depth=4, ccp_alpha=0),
        DecisionTreeClassifier(max_depth=5, ccp_alpha=0),
        DecisionTreeClassifier(max_depth=6, ccp_alpha=0),
        DecisionTreeClassifier(max_depth=7, ccp_alpha=0),
        DecisionTreeClassifier(max_depth=10, ccp_alpha=0),
        DecisionTreeClassifier(max_depth=3, ccp_alpha=0.005),
        DecisionTreeClassifier(max_depth=4, ccp_alpha=0.005),
        DecisionTreeClassifier(max_depth=5, ccp_alpha=0.005),
        DecisionTreeClassifier(max_depth=6, ccp_alpha=0.005),
        DecisionTreeClassifier(max_depth=7, ccp_alpha=0.005),
        DecisionTreeClassifier(max_depth=10, ccp_alpha=0.005),
        DecisionTreeClassifier(max_depth=3, ccp_alpha=0.01),
        DecisionTreeClassifier(max_depth=4, ccp_alpha=0.01),
        DecisionTreeClassifier(max_depth=5, ccp_alpha=0.01),
        DecisionTreeClassifier(max_depth=6, ccp_alpha=0.01),
        DecisionTreeClassifier(max_depth=7, ccp_alpha=0.01),
        DecisionTreeClassifier(max_depth=10, ccp_alpha=0.01),
        DecisionTreeClassifier(max_depth=3, ccp_alpha=0.02),
        DecisionTreeClassifier(max_depth=4, ccp_alpha=0.02),
        DecisionTreeClassifier(max_depth=5, ccp_alpha=0.02),
        DecisionTreeClassifier(max_depth=6, ccp_alpha=0.02),
        DecisionTreeClassifier(max_depth=7, ccp_alpha=0.02),
        DecisionTreeClassifier(max_depth=10, ccp_alpha=0.02),
        DecisionTreeClassifier(max_depth=3, ccp_alpha=0.03),
        DecisionTreeClassifier(max_depth=4, ccp_alpha=0.03),
        DecisionTreeClassifier(max_depth=5, ccp_alpha=0.03),
        DecisionTreeClassifier(max_depth=6, ccp_alpha=0.03),
        DecisionTreeClassifier(max_depth=7, ccp_alpha=0.03),
        DecisionTreeClassifier(max_depth=10, ccp_alpha=0.03),
    ],
    "proto_selection":[
        ClusterCentroids(sampling_strategy={0: 3, 1: 3}),
        ClusterCentroids(sampling_strategy={0: 5, 1: 5}),
        ClusterCentroids(sampling_strategy={0: 7, 1: 7}),
        ClusterCentroids(sampling_strategy={0: 9, 1: 9}),
    ],
    "unbalanced_rate":[0.1, 0.2],
    "minimum_regions": [2],
    "min_support": [100, 300, 500, 1000],
    "n_jobs": [6]
}
additional_params = {
    "max_depth":(getMaxDepth,"base_estimator"),
    "ccp_alpha":(getCcpAlpha,"base_estimator"),
    "n_prototypes":(getNPrototypes, "proto_selection")
}
estimator = apd.APD_Classifier()
search = GridSearch(estimator, param_grid,mlflow_scoring_wrapper,1,10, ml_flow=mlFlow, additional_params=additional_params)
for file in files:
    path = f"{folder}\\{file}.csv"
    try:
        data = pd.read_csv(path, sep=",")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue
    mlFlow.add_dataset(data, path, file)
    ohe = LabelEncoder()
    cols = [col for col in data.columns if col not in ["LABEL", "id", "id.1", "id.2", "id.3"]]
    X = data.loc[:, cols].values
    y = data.loc[:, "LABEL"].values
    y = ohe.fit_transform(y)
    mlFlow.rename_experiment(f"DT_{file}")
    search.fit(X,y)
