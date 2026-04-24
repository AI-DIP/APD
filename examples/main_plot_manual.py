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

from adversarial_prototype_decomposition.base import apd as apdlib
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import addcopyfighandler

mpl.use("QtAgg")

def plotData(x, y, label1, label2=None, colors='rgb', markers=['.', '.'], markersize=5):
    print(df1.a1.shape)
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
df1 = pd.read_csv('examples\\Data\\Results\\train_regions_1.csv', sep=";")
# df1 = pd.read_csv('examples\\Data\\Results\\banana.csv', sep=",")
#df1 = pd.read_csv("Data/banana.csv", sep = ",")
#df1.columns = ["a1","a2","Class"]
# df2 = pd.read_csv('Data/Results/proto_regions.csv',sep=";")
proto_type = "CC"
#proto_type = "manual"
#proto_type = "SAMPLE"
df11 = df1.copy()
do_voronoi = False
# df1 = df1.sample(500,axis=0)

width, height = 8, 6

# ux_protoPairs = np.unique(df1['ID_Proto_Pair'])

X = df11[["a1", "a2"]]
df11.loc[df11['Class']==-1, 'Class']=0
y = df11[['Class']].values

mi = np.min(X, axis=0)
mx = np.max(X, axis=0)
limx = (mi.a1, mx.a1)
limy = (mi.a2, mx.a2)


id1 = y == 1
n = 2
if proto_type=="SAMPLE":
    idx1 = df11[id1].sample(n=3,random_state=2001).index
    idx2 = df11[~id1].sample(n=2,random_state=2001).index

    PX,PY = (pd.concat((df11.loc[idx1, ["a1", "a2"]], df11.loc[idx2, ["a1", "a2"]])),  # df2[["a1","a2"]]
             pd.concat((df11.loc[idx1, ["Class"]],    df11.loc[idx2, ["Class"]])))  # df2[["a1","a2"]]
elif proto_type=="CC":
    PX,PY = ClusterCentroids(estimator=KMeans(random_state=0, n_init=10),
                             sampling_strategy={0: 3, 1: 3}).fit_resample(X,y)
    PY = pd.DataFrame(PY,columns=["Class"])

elif proto_type=="manual":
    PX, PY =(pd.DataFrame([[0.99470783, 0.51770136],[0.02465749, 0.14812839],[0.61552881, 0.10683872],[0.16161304, 0.94875707],[0.75546858, 0.68521314]], columns=["a1","a2"]).reset_index(drop=True),
             pd.DataFrame([[ 1.],[ 1.],[ 1.],[0.],[0.]], columns=["Class"]).reset_index(drop=True))


apd = apdlib.APD2(proto=PX,
                  proto_labels=PY,
                  unbalanced_rate=0.05,
                  min_support=100,
                  prune_regions=False,
                  minimum_n_regions=1)
# #    apdlib.APD(P, PY,unbalanced_rate=0, min_support=1))
regions = apd.generate_regions(X, y)
ux_protoPairs = list(regions.keys())
stats = apd.region_stats

print(f"UX Proto F0:")
print(ux_protoPairs)
print("Counts")
print(stats)
print("==========")

q = apd.assign_regions(X, ux_protoPairs)

xlist = np.linspace(limx[0], limx[1], 100)
ylist = np.linspace(limy[0], limy[1], 100)
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
PY = pd.DataFrame(apd.proto_labels[protos_id,:], columns=["Class"])

qcc = np.reshape(qcc, Xc.shape)

n = len(ux_protoPairs)
n += 2  # Reserwujemy dodatkowe dwa kolory na klasy

colors = mpl.colormaps[
    # 'tab20'
    "gist_ncar"
].resampled(n)

#%%
cols = colors(range(n + 1))


cols2 = cols[[0, n]]
cols = cols[1:n]
cols2[1][0]=0.5
cols2[1][1]=0.5

# colors2 = cols[0:n-1]
# colors = cols[2:n]


plt.figure(1, figsize=(width, height))
plt.clf()
plotData(df1.a1, df1.a2, label1=df1["Class"],
         markers=['o'], colors=cols2, markersize=4)
plotData(PX.a1, PX.a2, PY.Class, markers=['*', 'o'], colors='rr', markersize=15)

if isinstance(apd, apdlib.APD2_MIDDLE_POINT):
    plt.plot(apd.pair_center[:,0],apd.pair_center[:,1],color="white",marker="+",linestyle="None", markersize=20)
if isinstance(apd, apdlib.APD2_MIDDLE_POINT):
    if(apd.removed_pair_center is not None and len(apd.removed_pair_center) != 0):
        plt.plot(apd.removed_pair_center[:,0],apd.removed_pair_center[:,1],color="black",marker="+",linestyle="None", markersize=20)
PX.reset_index(inplace=True, drop=True)
protos_id_to_row = dict(zip(protos_id,range(len(protos_id)))) #Mapowanie proto_id na numer wiersza
for pair in ux_protoPairs:
    i, j = apd.unpairCantor(pair)
    x = PX.loc[[protos_id_to_row[i], protos_id_to_row[j]], "a1"]
    y = PX.loc[[protos_id_to_row[i], protos_id_to_row[j]], "a2"]
    plt.plot(x, y, 'r')
ax = plt.gca()
p = PX  # = df2[["a1","a2"]].values
# p = np.vstack([p, [[0, 1],[1, 0]]])
if do_voronoi:
    vor = Voronoi(p)
    voronoi_plot_2d(vor,
                    ax=ax,
                    show_points=False,
                    show_vertices=False)


cp = plt.contourf(Xc, Yc, qcc, alpha=0.7, cmap="gist_ncar")  # colors=cols)
plt.show()
plt.xlim(limx)
plt.ylim(limy)
plt.show()
