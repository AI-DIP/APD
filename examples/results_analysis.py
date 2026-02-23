import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
mpl.use("QtAgg")
df = pd.read_excel("Data/Results/results_apd_tree-10+15+20.xlsx")
df = df.sort_values(by="depth")
ds = np.unique(df["dataset"])
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 26}

name_mapper = {"RF":"Random Forest",
               "APD2": "PTD",
               "Tree":"Tree"}

mpl.rc('font', **font)
#%%
for i,d in enumerate([d for d in ds]):
    print(d)
    id = df.dataset == d
    dfs = df.loc[id,:]
    plt.figure(i, clear=True, figsize=(10,8))
    #plt.title(d)
    idm = dfs["model"] == "RF2"
    acc_rf = dfs.loc[idm,"me"].values[-1]
    depth_rf = [2,10]
        #dfs.loc[idm, "depth"].values)[[0,-1]]
    #plt.plot(depth_rf,[acc_rf,acc_rf],":k", label="Random Forest")
    #plt.plot([acc_rf, acc_rf], depth_rf, ":k", label="Random Forest")


    for m in ["APD","Tree"]:
        idm = dfs["model"]==m
        me = dfs.loc[idm,"me"]
        de = dfs.loc[idm,"depth"]
        label = name_mapper[m]
        if "APD" in m:
            label = label + "(" + str(int(np.mean(dfs.loc[idm,"regions"]))) + ")"
        #plt.plot(de,me, label=label)
        plt.plot( de,me, label=label)
    #plt.ylim((np.min(dfs["me"]) - 0.5, np.max(dfs["me"])+0.5))
    #plt.xlim((2,10))
    plt.ylim((np.min(dfs["me"]) - 0.5, np.max(dfs["me"]) + 0.5))
    plt.xlim((2,10))
    plt.ylabel("Accuracy [-]")
    plt.xlabel("Tree depth [-]")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(f'Data/Results/pic/pic_inv_{d}.png', dpi=100)