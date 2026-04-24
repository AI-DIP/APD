import pandas as pd
import os
import mlflow

from mlflow.data.pandas_dataset import from_pandas
from itertools import product

EXP_NAME = "SVM_TEST3"

dataDir = 'Y:\\Datasets\\datasets_ppe\\'
datasets = [
    # (dataDir,"codrnaNorm"),
    (dataDir,"electricity-normalized"),
    # (dataDir,"covtype"),
    # (dataDir, "Agrawal1"),
    # (dataDir, "shuttle2"),
    (dataDir, "banana"),
    (dataDir, "coil2000"),
    (dataDir, "magic"),
    (dataDir, "phoneme"),
    (dataDir, "ring"),
    # (dataDir, "spambase"),
    (dataDir, "twonorm"),
    # (dataDir, "titanic"),
]

def login():
    if os.environ.get("MLFLOW_TRACKING_USERNAME",None) is None or os.environ.get("MLFLOW_TRACKING_PASSWORD",None) is None:
        # user = input("USER: ")
        # password = input("PASSWORD: ")
        user = "admin"
        password = "passwor"
        os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()

def create_run(setup:dict, datasets:dict, name_field:list=["MODEL"]):
    keys = setup.keys()
    values = setup.values()

    permutation = [dict(zip(keys,v)) for v in product(*values)]
    for p in permutation:
        run_name = "_".join(str(p[f]) for f in name_field)
        with mlflow.start_run(run_name=run_name):
            dataset = from_pandas(datasets["dataset"], name=datasets["name"], source=datasets["path"])
            mlflow.log_input(dataset)
            for k in p:
                if isinstance(p[k], dict):
                    pkeys = p[k].keys()
                    pvalues = p[k].values()
                    ppermutation = [dict(zip(pkeys,pv)) for pv in product(*pvalues)]
                    for pp in ppermutation:
                        for pk in pp:
                            mlflow.set_tag(f"{k}_{pk}",pp[pk])
                else:
                    mlflow.set_tag(k, p[k])
            mlflow.set_tag("STATUS", "WAITING")
            mlflow.end_run()
    # name = "_".join(s for s in setup.get(na))

login()

mlflow.set_tracking_uri("http://192.168.10.40:5000")
mlflow.set_experiment(EXP_NAME)

SVM = {
    "MODEL":["SVM"],
    "SVM_C":[0.1],
    "SVM_KERNEL":['rbf'],
    "SVM_CACHE_SIZE":[2048]
}

CSVM = {
    "MODEL":["CSVM"],
    "CSVM_C":[0.1],
    "CSVM_KERNEL":['rbf'],
    "CSVM_CACHE_SIZE":[2048]
}

APD = {
    "MODEL":["APD"],
    "APD_ESTIMATOR": [SVM],
    "APD_PROTO_SELECTION":["K-MEANS", "GLVQ", "GMLVQ"],
    "APD_N_PROTO": [3,5,7,9],
    "APD_MIN_SUPPORT": [100,500,1000],
    "APD_UNBALANCED_RATE": [0.1,0.2,0.3]
}


for dirName, fName in datasets:
    path = dirName+fName+".csv"
    df = pd.read_csv(path, sep=";", quoting=1, quotechar='"')
    if(df.shape[0] < 50000):
        create_run(SVM,datasets={"dataset":df, "name":fName, "path":path}, name_field=["MODEL"])
    create_run(APD,datasets={"dataset":df, "name":fName, "path":path}, name_field=["MODEL", "APD_PROTO_SELECTION", "APD_N_PROTO", "APD_MIN_SUPPORT"])
    # create_run(CSVM,datasets={"dataset":df, "name":fName, "path":path}, name_field=["MODEL"])

    # print(f"[{fName}] {df.shape[0]}")
