import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("QtAgg")
name_mapper = {"RF":"Random Forest",
               "APD2": "PTD",
               "Tree":"Tree"}
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 26}

mpl.rc('font', **font)

#%%
df = pd.read_excel("Data/Results/results_apd3_tree-40.xlsx")
df = df.sort_values(by=["depth","n_clusters"])
ds = np.unique(df["dataset"])
#%%
#ds = ds[0:3]
param_col = "min_support"
x_col = 'regions'
params = np.unique(df[param_col])

#%%

for i,d in enumerate([d for d in ds]):
    print(d)
    id = df.dataset == d
    dfs = df.loc[id,:]
    plt.figure(i, clear=True, figsize=(10,8))
    #plt.title(d)



    for param in params:
        idm = dfs[param_col] == param
        me = dfs.loc[idm,"me"]
        de = dfs.loc[idm,x_col]
        label = name_mapper["APD"] + "(" + str(param) + ")"
        # if "APD" in m:
        #     label = label + "(" + str(int(np.mean(dfs.loc[idm,"regions"]))) + ")"
        #plt.plot(de,me, label=label)
        plt.plot(de, me, label=label)
    plt.ylim((np.min(dfs["me"]) - 0.5, np.max(dfs["me"]) + 0.5))
    #plt.xlim((2,10))
    plt.ylabel("Accuracy [-]")
    plt.xlabel("# regions [-]")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(f'Data/Results/pic/pic_param_{d}.png', dpi=100)