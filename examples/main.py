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
from src.adversarial_prototype_decomposition.sampler.kmeans_sampler import SimpleClusterCentroids
from imblearn.under_sampling import ClusterCentroids
from sklearn.pipeline import Pipeline
from src.adversarial_prototype_decomposition.classifier import classifiers as  apd
from sklearn.base import clone
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer


cv = StratifiedKFold(random_state=1, n_splits=10, shuffle=True)

dataDir = 'D:\\Projects\\DataMining\\Data\\CSV_Experiments\\Above_1k\\'
dataDirLarge = 'D:\\Projects\\DataMining\\Data\\Large - OpenML\\'
datasets = [
    (dataDirLarge,"codrnaNorm"),
    (dataDirLarge,"electricity-normalized"),
    (dataDirLarge,"covtype"),
    (dataDir, "banana"),
    (dataDir, "ring"),
    (dataDir, "twonorm"),
]

all_res = []
tts = []
for max_depth in [3,4,5,6,7,10]:
    base_estimator = DecisionTreeClassifier(max_depth=max_depth) #RandomForestClassifier()
    ite = 1
    models = [
            ("APD2", apd.APD_Classifier(apd_type="apd2",
                                        base_estimator=clone(base_estimator),
                                        #proto_selection=ClusterCentroids(sampling_strategy={-1:5,1:5}),
                                        proto_selection=ClusterCentroids(sampling_strategy={0: 10, 1: 10}),
                                        unbalanced_rate=0.2,
                                        minimum_regions=2,
                                        min_support=400,
                                        n_jobs=6)),

                ("Tree", clone(base_estimator)),
                ]

    for dirName,fName in datasets:
        #df = pd.read_csv(dirName + fName + "\\" + fName + ".dat",sep=";")
        df = pd.read_csv(dirName + fName + ".csv", sep=";",quoting=1, quotechar='"')
        X = df[[column for column in df.columns if column not in ["LABEL","id"]]]
        y = np.squeeze(df[['LABEL']].values)
        ohe = LabelEncoder()
        y = ohe.fit_transform(y)
        ohe = OneHotEncoder()
        sym_cols = (X.dtypes == "O").values
        ct = ColumnTransformer([('ohe',ohe, sym_cols)],remainder = 'passthrough')
        X = ct.fit_transform(X)

        print("CV")
        #scores = {}
        scores=[]
        for model_name, model in models:
            score = cross_val_score(clone(model), X, y, cv=cv)
            #scores[model_name] = score
            me = np.mean(score) * 100
            st = np.std(score) * 100
            print(f"{model_name}: {[me, st]}")
            res = {"model": model_name,
                   "dataset": fName,
                   "me":me,
                   "std":st,
                   "depth": max_depth
                   }
            if type(model)==apd.APD_Classifier:
                res["regions"] = model.region_stats.shape[0]
            all_res.append(res)
            scores.append((model_name,score))
        model_name1,d1 = scores[0]
        pvs = []
        for i in range(1, len(scores)):
            model_name2, d2 = scores[i]
            tt = stats.ttest_rel(d1, d2)
            pvs.append(
                {"refName":model_name1,
                 "modName":model_name2,
                 "p-value":tt.pvalue,
                 "reject":tt.pvalue < 0.05,
                 "max_depth":max_depth}
            )
        tts.append(pvs)


df_res = pd.DataFrame(all_res)
df_tt = pd.DataFrame(tts)
df_res.to_excel(f"Data/Results/results_apd_tree-{ite}.xlsx")
df_res.to_excel(f"Data/Results/results_apd_tree_tt-{ite}.xlsx")

