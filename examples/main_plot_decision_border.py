# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:08:25 2023

@author: Marcin
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.under_sampling import ClusterCentroids
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cc
#import sklearn_extra.cluster as cce
from src.adversarial_prototype_decomposition.base import apd as ppelib
from src.adversarial_prototype_decomposition.classifier import classifiers as ppec
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import addcopyfighandler

#mpl.use("QtAgg")

def plotData(x, y, label1, label2=None, colors='rgb', markers=['.', '.'], markersize=3):
    #print(df1.a1.shape)
    uxs = np.unique(label1)
    lux = len(uxs)
    lux2 = 1

    if label2 is not None:
        uxs2 = np.unique(label2)
        lux2 = len(uxs2)

    if len(markers) == 1:
        markers = markers * (lux * lux2)

    for i, ux in enumerate(uxs):
        id1 = label1 == ux
        if label2 is not None:
            for j, ux2 in enumerate(uxs2):
                id2 = label2 == ux2
                id12 = id1 & id2
                print(np.sum(id12))
                plt.plot(x[id12], y[id12], color=colors[j], marker=markers[i], linestyle="None", markersize=markersize)
        else:
            print(np.sum(id1))
            plt.plot(x[id1], y[id1], color=colors[i], marker=markers[i], linestyle="None", markersize=markersize)




fName = "poly"
df1 = pd.read_csv('Data/Results/train_regions.csv', sep=";")
df1 = pd.read_csv("Data/banana.csv", sep = ",")
df1 = pd.read_csv("Data/sin.csv", sep = ",")
df1 = pd.read_csv("Data/4_clust.csv", sep = ",")
#df1.columns = ["a1","a2","Class"]

# df2 = pd.read_csv('Data/Results/proto_regions.csv',sep=";")
proto_type = "CC_Banana"
#proto_type = "manual"
#proto_type = "SAMPLE"
df11 = df1.copy()
do_voronoi = False
soSave = False
# df1 = df1.sample(500,axis=0)

width, height = 8, 6

# ux_protoPairs = np.unique(df1['ID_Proto_Pair'])

X = df11[["a1", "a2"]]
df11.loc[df11['Class']==-1, 'Class']=0
y = df11[['Class']].values

pr = StandardScaler()

X = pr.fit_transform(X)

mi = np.min(X, axis=0)
mx = np.max(X, axis=0)
#limx = (mi.a1, mx.a1)
#limy = (mi.a2, mx.a2)

limx = (mi[0], mx[1])
limy = (mi[0], mx[1])


id1 = y == 1
n = 2
metric = "sqeuclidean"#"squeuclidian"#'cityblock'
model : ppec.APD_Classifier = ppec.APD_Classifier(
                   #type="pe",
                   apd_type="apd2",
                   base_estimator=DecisionTreeClassifier(max_depth=1,min_samples_leaf=3),
                   # proto_selection=ClusterCentroids(sampling_strategy={-1:5,1:5}),
                   proto_selection=ClusterCentroids(sampling_strategy={0: 4, 1: 2}),#, estimator=cce.KMedoids(init="build")),
                   #proto_selection=ClusterCentroids(sampling_strategy={0: 6, 1: 4}),#, estimator=cce.KMedoids(init="build")),
                   unbalanced_rate=0.01,
                   minimum_regions=2,
                   min_support=100,
                   n_jobs=6,
                   metric= metric #'chebyshev'
                    )
model.fit(X,y)
PX = model.proto_ensemble_.proto
PY = model.proto_ensemble_.proto_labels
# PX,PY = ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
#                          sampling_strategy={0: 10, 1: 10}).fit_resample(X,y)
PY = pd.DataFrame(PY,columns=["Class"])



apd = model.proto_ensemble_
regions = apd.generate_regions(X, y)

ux_protoPairs = list(regions.keys())
stats = apd.region_stats

print(f"UX Proto F0:")
print(ux_protoPairs)
print("Counts")
print(stats)
print("==========")

q = apd.assign_regions(X, ux_protoPairs)

xlist = np.linspace(limx[0], limx[1], 200)
ylist = np.linspace(limy[0], limy[1], 200)
Xc, Yc = np.meshgrid(xlist, ylist)

yc = np.reshape(Yc, (-1, 1))
xc = np.reshape(Xc, (-1, 1))
xyc = np.concatenate((xc, yc), axis=1)

qc = apd.assign_regions(xyc, ux_protoPairs)
qcc = np.zeros((xyc.shape[0],1))
for i,(k,v) in enumerate(qc.items()):
    qcc[v]=i


protos_id = np.array(sorted(set(sum(map(apd.unpairCantor, ux_protoPairs), ()))))
PX = pd.DataFrame(apd.proto[protos_id,:],columns=["a1","a2"])
PY = pd.DataFrame(apd.proto_labels[protos_id], columns=["Class"])

qcc = np.reshape(qcc, Xc.shape)

n = len(ux_protoPairs)
n += 2  # Reserwujemy dodatkowe dwa kolory na klasy


#%%
#plotData(X[:,0], X[:,1], label1=df1["Class"],
#         markers=['o'], colors=cols2, markersize=1)
#plotData(PX.a1, PX.a2, PY.Class, markers=['*', 'o'], colors='rr', markersize=15)
PX.reset_index(inplace=True, drop=True)
protos_id_to_row = dict(zip(protos_id,range(len(protos_id)))) #Mapowanie proto_id na numer wiersza

dcc = model.predict(xyc)
dcc = np.reshape(dcc, Xc.shape)
p = PX

#%%
colors = mpl.colormaps[
    # 'tab20'
    #"gist_ncar"
    "Set1"
].resampled(n)

cols = colors(range(n + 1))


cols2 = cols[[0, n]]
cols = cols[1:n]
cols2[1][0]=0.5
cols2[1][1]=0.5

# colors2 = cols[0:n-1]
# colors = cols[2:n]

plt.figure(1, figsize=(width, height))
plt.clf()
for pair in ux_protoPairs:
    i, j = apd.unpairCantor(pair)
    x1 = PX.loc[[protos_id_to_row[i], protos_id_to_row[j]], "a1"]
    x2 = PX.loc[[protos_id_to_row[i], protos_id_to_row[j]], "a2"]
    plt.plot(x1, x2, 'r')
ax = plt.gca()
 # = df2[["a1","a2"]].values
# p = np.vstack([p, [[0, 1],[1, 0]]])
if do_voronoi:
    vor = Voronoi(p)
    voronoi_plot_2d(vor,
                    ax=ax,
                    show_points=False,
                    show_vertices=False)

cp = plt.contourf(Xc, Yc, dcc, alpha=0.7, cmap="Dark2")#"gist_ncar")  # colors=cols)
cp = plt.contour(Xc, Yc, qcc, alpha=0.7)  # colors=cols)

plt.scatter(X[:,0], X[:,1], c=y,marker='o',  s=30, cmap="Paired")
idC1 = PY.Class==1
idC2 = PY.Class!=1
plt.scatter(PX.a1[idC1], PX.a2[idC1], c='r', marker='*', s=200)
plt.scatter(PX.a1[idC2], PX.a2[idC2], c='r', marker='o', s=100)

# plt.colormap(hot)
plt.xlim(limx)
plt.ylim(limy)
if soSave:
    plt.savefig(f'pic/local_ppd_scatter.png', bbox_inches='tight')

for i,id in enumerate(model.fitted_base_models_):
    plt.figure(10+i,clear=True)
    plot_tree(model.fitted_base_models_[id])
    if soSave:
        plt.savefig(f'pic/local_ppd_tree_{i}.png', bbox_inches='tight')

#%%
plt.figure(20,clear=True)
model_ref = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
model_ref.fit(X, y)
plot_tree(model_ref)
if soSave:
    plt.savefig(f'pic/local_tree_single.png', bbox_inches='tight')

plt.figure(50, clear=True)
dcc_ref = model_ref.predict(xyc)
dcc_ref = np.reshape(dcc_ref, Xc.shape)
cp = plt.contourf(Xc, Yc, dcc_ref, alpha=0.7, cmap="Dark2")#"gist_ncar")  # c# olors=cols)
plt.scatter(X[:,0], X[:,1], c=y,marker='o',  s=30, cmap="Paired")
if soSave:
    plt.savefig(f'pic/local_tree_scatter.png', bbox_inches='tight')
