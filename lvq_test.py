#%%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklvq import GLVQ
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path_train_data = "Y:/DDabrowski/Dataset/Prepared_UT1_v5_048.csv"
train_data = pd.read_csv(path_train_data, sep=",")
ohe = LabelEncoder()
cols = [col for col in train_data.columns if col not in ["LABEL", "id", "id.1", "id.2", "id.3", "Applied torque"]]
X_train = train_data.loc[:, cols].values
feature_names = cols
y_train = train_data.loc[:, "LABEL"].values
y_train = ohe.fit_transform(y_train)
class_names = ["0","1"]

path_test_data1 = "Y:/DDabrowski/Dataset/Prepared_UT2_v5.csv"
test1_data = pd.read_csv(path_test_data1, sep=",")
path_test_data2 = "Y:/DDabrowski/Dataset/Prepared_UT3_v5.csv"
test2_data = pd.read_csv(path_test_data2, sep=",")
test_data = pd.concat([test1_data, test2_data])
X_test = test_data.loc[:, cols].values
y_test = test_data.loc[:, "LABEL"].values
y_test = ohe.transform(y_test)

class ProcessLogger:
    def __init__(self):
        self.states = np.array([])

    # A callback function has to accept two arguments, i.e., model and state, where model is the
    # current model, and state contains a number of the optimizers variables.
    def __call__(self, state):
        self.states = np.append(self.states, state)
        return False  # The callback function can also be used to stop training early,
        # if some condition is met by returning True.
logger = ProcessLogger()
scaler = StandardScaler()
X = scaler.fit_transform(X_train)
Xt = scaler.transform(X_test)
#%%
pn = 10
n_proto = [pn,pn]
model = GLVQ(prototype_n_per_class=np.array(n_proto),
             solver_params={"step_size": 0.1,
                            "max_runs": 100,
                            "batch_size": 16,
                            # "callback": logger,
                            })
model.fit(X,y_train)
acc = accuracy_score(y_true=y_train, y_pred=model.predict(X))
print(f"Accuracy: {acc}")

#%%

res_all ={}

protos_per_class = [4,5,6,7,8,9,10]#[3]#[5,10,20,40,80]
BATCH_SIZES = [128] #@[16,128,256]:
MAX_ITERS = [300]
plt.figure(31, clear=True)
for mr in MAX_ITERS:
    for bs in BATCH_SIZES:
        for ss in [0.1]:
            res = {"acc_train": [],
                   "acc_test": [],
                   "protos": [],
                   "acc_ext": []}
            for pn in protos_per_class:
                n_proto = [pn,pn]
                logger = ProcessLogger()
                # model = GLVQ(prototype_n_per_class=np.array(n_proto),
                #              solver_params={"step_size": ss,
                #                             "max_runs": mr,
                #                             "batch_size": bs,
                #                             #"callback": logger,
                #                             })
                #model = RandomForestClassifier()
                model = DecisionTreeClassifier(max_depth=pn)
                #model.fit(X, y_train)
                res_cur = cross_validate(estimator=model, X=X, y=y_train, cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring="accuracy", return_train_score=True)
                res_all[f"batch={bs} step={ss} iter={mr} proto={pn}"] = res_cur

                res["protos"].append(pn)
                res["acc_train"].append(np.mean(res_cur["train_score"]))
                res["acc_test"].append(np.mean(res_cur["test_score"]))
                model.fit(X,y_train)
                acc = model.score(Xt,y_test)
                res["acc_ext"].append(acc)
                print(f" Proto:{pn} | CV Results:{np.mean(res_cur['test_score'])} acc_Ut1+Ut3:{acc}")
            plt.plot(res["protos"],res["acc_train"],label=f"train {bs} {ss} {mr}")
            plt.plot(res["protos"],res["acc_test"],label=f"test  {bs} {ss} {mr}")
plt.legend()
plt.show()
#%%
for bs in BATCH_SIZES:
    for it in MAX_ITERS:
        for pn in protos_per_class:
            key = f'batch={bs} step=0.1 iter={it} proto={pn}'
            acc = np.mean(res_all[key]['test_score'])
            print(key + f"  acc = {acc}")
#%%
model = GLVQ(prototype_n_per_class=np.array(n_proto),
             solver_params={"step_size": 0.1,
                            "max_runs": 300,
                            "batch_size": 128,
                            "callback": logger,
                            })
model.fit(X,y_train)
iteration, fun = zip(*[(state["nit"], state["fun"]) for state in logger.states])
plt.figure(2, clear=True)
plt.title("Learning Curve (Less is better)")
plt.plot(iteration, fun)
plt.show()

plt.figure(1, clear=True)
colX = [i for (i,col) in enumerate(cols) if col == 'Pressure - output']
colY = [i for (i,col) in enumerate(cols) if col == 'Flow - leak line']
x1 = X[:, colX]
x2 = X[:, colY]
plt.scatter(x1,x2,c=y_train)

Xp = model.prototypes_
yp = model.prototypes_labels_
xp1 = Xp[:, colX]
xp2 = Xp[:, colY]
plt.scatter(xp1,xp2,c="k",marker='x',s=100)

plt.show()

#%%
from visualization import tsne_classification_pipeline, pca_classification_pipeline, umap_classification_pipeline

Xd = np.vstack((X, Xp))
yd = np.hstack((y_train,2*np.ones((yp.shape))))
tsne_classification_pipeline(Xd,yd,["normal","f1","P"])
umap_classification_pipeline(Xd,yd,["normal","f1","P"])
pca_classification_pipeline(Xd,yd,["normal","f1","P"])
