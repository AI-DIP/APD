# %%0. Imports
## IMPORTS
import pandas as pd
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
import tempfile
import os
import mlflow
## FROM
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import export_text, plot_tree
from mlflow.data.pandas_dataset import from_pandas
#OUR
from adversarial_prototype_decomposition.sampler import GLVQ_Sampler
from adversarial_prototype_decomposition.classifier.classifiers import APD_ClassifierScaler, APD_Classifier
from adversarial_prototype_decomposition.utils.mlflow_utils import save_fig_as_artefact, save_pandas_as_artefact
from adversarial_prototype_decomposition.utils.apd_utils import get_proto_info, get_region_info
from pruned_decision_tree import PrunedDecisionTree

def login():
    if os.environ.get("MLFLOW_TRACKING_USERNAME",None) is None or os.environ.get("MLFLOW_TRACKING_PASSWORD",None) is None:
        # user = input("USER: ")
        # password = input("PASSWORD: ")
        user = "admin"
        password = "passwor"
        os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()

login()

#STATIC
##DATA
DATASET_LABEL = "LABEL"
MULTIPLY_BY_1000 = ["Sensor 1", "Sensor 2", "Sensor 3"]
EXCLUDE_METADATA = ["LABEL", "id", "id.1", "id.2", "id.3", "Applied torque"]
EXCLUDE_FLOW = ["Flow - leak line", "Flow - output"]
EXCLUDE_SENSEOR = ["Sensor 1", "Sensor 2", "Sensor 3"]
EXCLUDE_OTHER = ["Temp. diff"]
EXCLUDE = [*EXCLUDE_METADATA, *EXCLUDE_OTHER, *EXCLUDE_SENSEOR]
##RUN
TEST_RUN = False
USE_PREVIOUS_PROTO = True
APD_RUN = True
EXP_NAME= "PN_POMAR_11"

if TEST_RUN:
    EXP_NAME = "APD_TEST"
##PARAMS
CCP = [0.0]#,0.02,0.03]
N_PROTO = [3,5]#,7,9]
DT_MAX_DEPTH = [3,4,5]#,7,9]
MIN_SUPORTS = [500,1000,2000]
UNBALANCED_RATES = [0.2,0.3]
DIST_METRIC = ["sqeuclidean"]
PROTO_SELECTION = ["GLVQ", "KMEANS"]

if not APD_RUN:
    N_PROTO = [0]
    MIN_SUPORTS = [0]
    UNBALANCED_RATES = [0]
    PROTO_SELECTION = [None]

param_dict = {
    "PROTO_SELECTION": PROTO_SELECTION,
    "N_PROTO": N_PROTO,
    "CCP": CCP,
    "DT_MAX_DEPTH": DT_MAX_DEPTH,
    "MIN_SUPORTS": MIN_SUPORTS,
    "UNBALANCED_RATES": UNBALANCED_RATES,
    "DIST_METRIC": DIST_METRIC
}

# Generowanie permutacji
keys = list(param_dict.keys())
all_combinations = list(itertools.product(*(param_dict[key] for key in keys)))
list_of_dicts = [dict(zip(keys, values)) for values in all_combinations]

# %%1. Zaladuj dane
def multiply_columns(data:pd.DataFrame, columns:list, value:float=1000.0):
    for column in columns:
        data[column] *= value
    return data

path_train_data = "Y:/DDabrowski/Dataset/Prepared_UT1_v5_048.csv"
train_data = pd.read_csv(path_train_data, sep=",")
ohe = LabelEncoder()
train_data = multiply_columns(train_data, MULTIPLY_BY_1000)

#SELECT DATA
cols = [col for col in train_data.columns if col not in EXCLUDE]
feature_names = cols

#TRAIN SET
X_train = train_data.loc[:, cols].values
y_train = train_data.loc[:, DATASET_LABEL].values
y_train = ohe.fit_transform(y_train)
class_names = ["0","1"]

#TEST1 SET
path_test_data1 = "Y:/DDabrowski/Dataset/Prepared_UT2_v5.csv"
test1_data = pd.read_csv(path_test_data1, sep=",")
test1_data = multiply_columns(test1_data, MULTIPLY_BY_1000)
X_test1 = test1_data.loc[:, cols].values
y_test1 = ohe.transform(test1_data.loc[:, "LABEL"].values)

#TEST1 SET
path_test_data2 = "Y:/DDabrowski/Dataset/Prepared_UT3_v5.csv"
test2_data = pd.read_csv(path_test_data2, sep=",")
test2_data = multiply_columns(test2_data, MULTIPLY_BY_1000)
X_test2 = test2_data.loc[:, cols].values
y_test2 = ohe.transform(test2_data.loc[:, "LABEL"].values)

# %%2.Data Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train,y_train)
X_test1 = scaler.transform(X_test1)
X_test2 = scaler.transform(X_test2)

# %%3.TEST

##LAST RUN
def print_run_info(params:dict, run:int, max_runs:int):
    print(f"[{run}/{max_runs}] RUN")
    for k in params.keys():
        print(f"[{k}] {params[k]}")
last_n_proto = -1
last_proto_selection = ""
pX,py = None,None

for run_number,params in enumerate(list_of_dicts):
    print_run_info(params, run_number+1,len(list_of_dicts))
    ##DT PARAMS
    max_depth = params["DT_MAX_DEPTH"]
    ccp = params["CCP"]
    ##APD PARAMS
    proto_selection = params["PROTO_SELECTION"]
    n_proto = params["N_PROTO"]
    min_suports = params["MIN_SUPORTS"]
    unbalanced_rates = params["UNBALANCED_RATES"]
    dist_metric = params["DIST_METRIC"]
    ##OTHER
    exp_name = EXP_NAME
    run_name = None
    if APD_RUN:
        run_name = f"APD_PROTO_{n_proto}_MAXDEPTH_{max_depth}_CCP_{ccp}"
    else:
        run_name = f"DT_MAXDEPTH_{max_depth}_CCP_{ccp}"
    
    # --- Podlaczenie do mlflow ---
    mlflow.set_tracking_uri("http://192.168.10.40:5000")
    mlflow.set_experiment(exp_name)
    tmp_dir = tempfile.mkdtemp()

    with mlflow.start_run(run_name=run_name):
        # --- Rejestracja datasetów jako INPUT ---
        train_dataset_info = from_pandas(train_data, name="Prepared_UT1_v5_048", source=path_train_data)
        test_dataset1_info = from_pandas(test1_data, name="Prepared_UT2_v5", source=path_test_data1)
        test_dataset2_info = from_pandas(test2_data, name="Prepared_UT3_v5", source=path_test_data2)

        mlflow.log_input(train_dataset_info, context="training")
        mlflow.log_input(test_dataset1_info, context="testing")
        mlflow.log_input(test_dataset2_info, context="testing")
        ## Base Estimator
        clf = PrunedDecisionTree(max_depth=max_depth, random_state=42, ccp_alpha=ccp)
        estimator = clf
        
        ## Proto Configuration
        proto = None
        if (USE_PREVIOUS_PROTO 
            and last_n_proto == n_proto 
            and last_proto_selection == proto_selection
            and pX is not None 
            and py is not None):
            proto = (pX, py)
        elif proto_selection == "GLVQ":
            proto = GLVQ_Sampler(
                        prototype_n_per_class=np.array([n_proto,n_proto]),
                        solver_params={"step_size": 0.1,
                        "max_runs": 100,
                        "batch_size": 64,},random='kmeans',random_state=42
                    )
            if USE_PREVIOUS_PROTO:
                pX, py = proto.fit_resample(X_train,y_train)
                proto = (pX,py)
        elif proto_selection == "KMEANS":
            proto = ClusterCentroids(sampling_strategy={0: n_proto, 1: n_proto})
            if USE_PREVIOUS_PROTO:
                pX, py = proto.fit_resample(X_train,y_train)
                proto = (pX,py)
        
        last_n_proto = n_proto 
        last_proto_selection = proto_selection

        ## change if APD run
        if APD_RUN:
            mlflow.log_params(clf.get_params())
            estimator = APD_Classifier(
                            apd_type="apd2",
                            base_estimator= clf,
                            min_support=min_suports,
                            unbalanced_rate= unbalanced_rates,
                            metric=dist_metric,
                            proto_selection=proto
                        )
        
        mlflow.log_param("n_prototypes",n_proto)
        mlflow.log_params(estimator.get_params())
        mlflow.set_tag("PROTO",proto_selection)
        mlflow.set_tag("DATA","FLOW_NO_TEMP")

        ## CROSS VALIDATION
        kf = StratifiedKFold(n_splits=10, shuffle=False)
        acc_scores, bacc_scores, f1_scores, fit_times, test_times = [], [], [], [], []
        cv = cross_validate(estimator, X_train, y_train, cv=kf, scoring=["balanced_accuracy", 'accuracy', 'f1_macro'],n_jobs=10)

        for i in range(10):
            mlflow.log_metric("CV_ACC", cv["test_accuracy"][i], step=i)
            mlflow.log_metric("CV_BACC", cv["test_balanced_accuracy"][i], step=i)
            mlflow.log_metric("CV_F1", cv["test_f1_macro"][i], step=i)
            mlflow.log_metric("CV_Fit_Time",cv["fit_time"][i], step=i)
            mlflow.log_metric("CV_Test_Time",cv["score_time"][i], step=i)

        mlflow.log_metrics({
            "CV_ACC_mean": np.mean(cv["test_accuracy"]),
            "CV_ACC_std": np.std(cv["test_accuracy"]),
            "CV_BACC_mean": np.mean(cv["test_balanced_accuracy"]),
            "CV_BACC_std": np.std(cv["test_balanced_accuracy"]),
            "CV_F1_mean": np.mean(cv["test_f1_macro"]),
            "CV_F1_std": np.std(cv["test_f1_macro"]),
            "CV_Fit_Time_mean": np.mean(cv["fit_time"]),
            "CV_Fit_Time_std": np.std(cv["fit_time"]),
            "CV_Test_Time_mean": np.mean(cv["score_time"]),
            "CV_Test_Time_std": np.std(cv["score_time"])
        })

        ## Train Model
        start_time = time.time()                            
        estimator.fit(X_train,y_train)
        fit_time = time.time() - start_time
        mlflow.log_metric("fit_time", fit_time)
        if APD_RUN:
            mlflow.log_metric("n_regions", estimator.region_stats.shape[0])
            indexes = {}
            used_alphas = {"region":[],"ccp_alpha":[]}
            for region in pd.DataFrame(estimator.region_stats).values:
                indexes[region[0]] = len(indexes)
                mlflow.log_metric("regions_size", region[1]+region[2], indexes[region[0]])
                save_pandas_as_artefact(get_proto_info(estimator, cols, scaler),"prototypes.csv",tmp_dir)
                save_pandas_as_artefact(get_region_info(estimator),"regions.csv",tmp_dir)
                features_importance_list = []
                for index in indexes:
                    model = estimator.fitted_base_models_.get(index)
                    used_alphas["region"].append(indexes[index])
                    used_alphas["ccp_alpha"].append(model.get_ccp_alpha())
                    mlflow.log_metric("used_cpp_alpha",model.get_ccp_alpha(),step=indexes[index])
                    rules_text = export_text(model._tree_estimator, feature_names=feature_names, decimals=4)
                    mlflow.log_text(rules_text, f"Rules{indexes[index]}.txt")
                    features_importance_list.append(list(model.get_feature_importances()))
                    path = model.get_complexity_pruning_cost()
                    acc_path = model.get_pruning_accuracy()
                    ccp_alphas, impurities = path.ccp_alphas, path.impurities
                    acc_ccp_alphas, accs = acc_path["ccp_alphas"], acc_path["acc_mean"]
                    ###impurities
                    fig, ax = plt.subplots()
                    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
                    ax.set_xlabel("Efektywna wartość alpha")
                    ax.set_ylabel("Całkowita nieczystość liści")
                    ax.set_title("Całkowita nieczystość vs efektywna wartość współczynnika alpha dla zbioru treningowego")
                    mlflow.log_figure(fig, f"Rules{indexes[index]}ccp_alpha_impurities.png")
                    save_pandas_as_artefact(pd.DataFrame(path),f"Rules{indexes[index]}ccp_alpha_impurities.csv",tmp_dir)
                    #A##CC
                    fig, ax = plt.subplots()
                    ax.plot(acc_ccp_alphas[:-1], accs[:-1], marker="o", drawstyle="steps-post")
                    ax.set_xlabel("Efektywna wartość alpha")
                    ax.set_ylabel("Dokładność")
                    ax.set_title("Dokładność vs efektywna wartość współczynnika alpha dla zbioru treningowego")
                    mlflow.log_figure(fig, f"Rules{indexes[index]}ccp_alpha_acc.png")
                    save_pandas_as_artefact(pd.DataFrame(acc_path),f"Rules{indexes[index]}ccp_alpha_acc.csv",tmp_dir)
                save_pandas_as_artefact(pd.DataFrame(features_importance_list, columns=feature_names),"features_importance.csv",tmp_dir, index_name="Region")
            save_pandas_as_artefact(pd.DataFrame(used_alphas),f"used_alphas.csv",tmp_dir)
        else:
            mlflow.log_metric("used_ccp_alpha",estimator.get_ccp_alpha())
            features_importance_list = []
            rules_text = export_text(estimator._tree_estimator, feature_names=feature_names)
            path = estimator.get_complexity_pruning_cost()
            acc_path = estimator.get_pruning_accuracy()
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            acc_ccp_alphas, accs = acc_path["ccp_alphas"], acc_path["acc_mean"]
            ### impurities
            fig, ax = plt.subplots()
            ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
            ax.set_xlabel("Efektywna wartość alpha")
            ax.set_ylabel("Całkowita nieczystość liści")
            ax.set_title("Całkowita nieczystość vs efektywna wartość współczynnika alpha dla zbioru treningowego")
            mlflow.log_figure(fig, f"ccp_alpha_impurities.png")
            save_pandas_as_artefact(pd.DataFrame(path),"ccp_alpha_impurities.csv",tmp_dir)
            ###ACC
            fig, ax = plt.subplots()
            ax.plot(acc_ccp_alphas[:-1], accs[:-1], marker="o", drawstyle="steps-post")
            ax.set_xlabel("Efektywna wartość alpha")
            ax.set_ylabel("Dokładność")
            ax.set_title("Dokładność vs efektywna wartość współczynnika alpha dla zbioru treningowego")
            mlflow.log_figure(fig, f"ccp_alpha_acc.png")
            save_pandas_as_artefact(pd.DataFrame(acc_path),"ccp_alpha_acc.csv",tmp_dir)
            mlflow.log_text(rules_text, f"Rules.txt")
            features_importance_list.append(list(estimator.get_feature_importances()))
            save_pandas_as_artefact(pd.DataFrame(features_importance_list, columns=feature_names),"features_importance.csv",tmp_dir, index_name="Id")

        ## TEST 1 MODEL
        start_time = time.time()                            
        y_test1_pred = estimator.predict(X_test1)
        test1_time = time.time() - start_time
        mlflow.log_metric("testUT2_time", test1_time)
        df = pd.DataFrame()
        df["y_true"] = y_test1
        df["y_pred"] = y_test1_pred
        df["pred_correct"] = y_test1 == y_test1_pred
        save_pandas_as_artefact(df,"predict_result_UT2.csv",tmp_dir)

        ### Metryki
        bacc = balanced_accuracy_score(y_test1, y_test1_pred)
        acc = accuracy_score(y_test1, y_test1_pred)
        f1 = f1_score(y_test1, y_test1_pred,average='macro')
        mlflow.log_metrics({"ACC_UT2": acc, "BACC_UT2": bacc, "F1_UT2": f1})
        print("TEST UT2:")
        print({"ACC": acc, "BACC": bacc, "F1": f1})
        
        ## TEST 2 MODEL
        start_time = time.time()                            
        y_test2_pred = estimator.predict(X_test2)
        test2_time = time.time() - start_time
        mlflow.log_metric("testUT3_time", test2_time)
        df = pd.DataFrame()
        df["y_true"] = y_test2
        df["y_pred"] = y_test2_pred
        df["pred_correct"] = y_test2 == y_test2_pred
        save_pandas_as_artefact(df,"predict_result_UT3.csv",tmp_dir)

        # Metryki
        bacc = balanced_accuracy_score(y_test2, y_test2_pred)
        acc = accuracy_score(y_test2, y_test2_pred)
        f1 = f1_score(y_test2, y_test2_pred,average='macro')
        mlflow.log_metrics({"ACC_UT3": acc, "BACC_UT3": bacc, "F1_UT3": f1})
        print("TEST UT3:")
        print({"ACC": acc, "BACC": bacc, "F1": f1})