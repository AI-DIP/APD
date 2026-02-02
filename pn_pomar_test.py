#%%
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import mlflow
import tempfile
import os
import numpy as np
from apdlib import classifiers as  apd
from mlflow.data.pandas_dataset import from_pandas
from imblearn.under_sampling import ClusterCentroids

from apdlib.sampler.glvq_sampler import GLVQ_Sampler
from utils.plot_utils import get_plot_regions_centres, get_plot, get_prototypes_plot_MDS
from utils.mlflow_utils import save_fig_as_artefact, save_pandas_as_artefact
from utils.apd_utils import get_proto_info, get_region_info

TEST_RUN = False
APD_RUN = True
MDS_RUN = False
DRAW_DT_PLOT = False

CCP = [0,0.005,0.01,0.02,0.03]
N_PROTO = [5]
DT_MAX_DEPTH = [3,4,5,6,7,8,10]
EXP_NAME= "DT_RulesX"

def login():
    if os.environ.get("MLFLOW_TRACKING_USERNAME",None) is None or os.environ.get("MLFLOW_TRACKING_PASSWORD",None) is None:
        user = input("USER: ")
        password = input("PASSWORD: ")
        os.environ["MLFLOW_TRACKING_USERNAME"] = user.strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password.strip()

login()

# 1. Zaladuj dane
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
test_data = pd.concat([test1_data,test2_data])
X_test = test_data.loc[:, cols].values
y_test = test_data.loc[:, "LABEL"].values
y_test = ohe.transform(y_test)

for ccp in CCP:
    for nproto2 in N_PROTO:
        for max_depth in DT_MAX_DEPTH:
            n_proto =nproto2
            exp_name = "DT_TEST2"
            run_name = None
            if APD_RUN:
                run_name = f"APD_PROTO_{n_proto}_MAXDEPTH_{max_depth}_CCP_{ccp}"
            else:
                run_name = f"DT_MAXDEPTH_{max_depth}_CCP_{ccp}"
            if not TEST_RUN:
                exp_name = EXP_NAME
            # --- Podlaczenie do mlflow ---
            mlflow.set_tracking_uri("http://192.168.10.40:5000")
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=run_name):
                # --- Rejestracja datasetów jako INPUT ---
                train_dataset_info = from_pandas(train_data, name="Prepared_UT1_v5_048", source=path_train_data)
                test_dataset1_info = from_pandas(test1_data, name="Prepared_UT2_v5", source=path_test_data1)
                test_dataset2_info = from_pandas(test2_data, name="Prepared_UT3_v5", source=path_test_data2)
                test_dataset = from_pandas(test_data, name="Prepared_UT2_v5 + Prepared_UT3_v5")

                mlflow.log_input(train_dataset_info, context="training")
                mlflow.log_input(test_dataset1_info, context="testing")
                mlflow.log_input(test_dataset2_info, context="testing")
                mlflow.log_input(test_dataset, context="testing")
                # Chwilowy katalog
                tmp_dir = tempfile.mkdtemp()

                # Trening
                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, ccp_alpha=ccp)
                estimator = clf
                if APD_RUN:
                    mlflow.log_params(clf.get_params())
                    estimator = apd.APD_ClassifierScaler(
                        type="apd2",
                        base_estimator= clf,
                        min_support=100,
                        unbalanced_rate= 0.1,
                        proto_selection=GLVQ_Sampler(prototype_n_per_class=np.array([n_proto,n_proto]),
                                                     solver_params={"step_size": 0.1,
                                                                    "max_runs": 200,
                                                                    "batch_size": 128,})
                        #ClusterCentroids(sampling_strategy={0: n_proto, 1: n_proto}),
                    )
                    mlflow.log_param("n_prototypes",n_proto)
                mlflow.log_params(estimator.get_params())
                #%%
                estimator.fit(X_train, y_train)

                if APD_RUN:
                    mlflow.log_metric("n_regions", estimator.region_stats.shape[0])
                    indexes = {}
                    for region in pd.DataFrame(estimator.region_stats).values:
                        indexes[region[0]] = len(indexes)
                        mlflow.log_metric("regions_size", region[1]+region[2], indexes[region[0]])
                        save_pandas_as_artefact(get_proto_info(estimator, cols),"prototypes.csv",tmp_dir)
                        save_pandas_as_artefact(get_region_info(estimator),"regions.csv",tmp_dir)
                        save_fig_as_artefact(get_plot(estimator,"Proto in 2axis",X_test,y_test,cols,"Flow - leak line","Pressure - output"), "Proto2AxisPlusData",tmp_dir)
                        save_fig_as_artefact(get_plot(estimator,"Proto in 2axis",None,None,cols,"Flow - leak line","Pressure - output"), "Proto2Axis",tmp_dir)
                        save_fig_as_artefact(get_plot_regions_centres(estimator,"Region Center in 2axis",X_test,y_test,cols,"Flow - leak line","Pressure - output"), "RegionCenter2AxisPlusData",tmp_dir)
                        save_fig_as_artefact(get_plot_regions_centres(estimator,"Region Center in 2axis",None,None,cols,"Flow - leak line","Pressure - output"), "RegionCenter2Axis",tmp_dir)
                        if MDS_RUN:
                            save_fig_as_artefact(get_prototypes_plot_MDS(estimator,"MDS"),"MDS",tmp_dir)
                        features_importance_list = []
                        for index in indexes:
                            model = estimator.fitted_base_models_.get(index)
                            rules_text = export_text(model, feature_names=feature_names, decimals=4)
                            mlflow.log_text(rules_text, f"Rules{indexes[index]}.txt")
                            features_importance_list.append(list(model.feature_importances_))
                            # Wizualizacja drzewa
                            if DRAW_DT_PLOT:
                                fig = plt.figure(figsize=(48, 24))
                                plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8, precision=4)
                                fig.tight_layout(pad=3.0)
                                fig.savefig("drzewo_deczyjne_czytelne.png", dpi=600, bbox_inches='tight', pad_inches=1.0)
                                mlflow.log_figure(fig, f"RulesGraph{indexes[index]}.png")
                        save_pandas_as_artefact(pd.DataFrame(features_importance_list, columns=feature_names),"features_importance.csv",tmp_dir, index_name="Region")
                else:
                    features_importance_list = []
                    rules_text = export_text(estimator, feature_names=feature_names)
                    mlflow.log_text(rules_text, f"Rules.txt")
                    # Wizualizacja drzewa
                    fig = plt.figure(figsize=(48, 24))
                    plot_tree(estimator, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8, precision=4)
                    fig.tight_layout(pad=3.0)
                    fig.savefig("drzewo_deczyjne_czytelne.png", dpi=600, bbox_inches='tight', pad_inches=1.0)
                    mlflow.log_figure(fig, f"RulesGraph.png")
                    features_importance_list.append(list(estimator.feature_importances_))
                    save_pandas_as_artefact(pd.DataFrame(features_importance_list, columns=feature_names),"features_importance.csv",tmp_dir, index_name="Id")
                # Predykcja
                y_pred = estimator.predict(X_test)
                df = pd.DataFrame()
                df["y_true"] = y_test
                df["y_pred"] = y_pred
                df["pred_correct"] = y_test == y_pred
                df["source_file"] = ['Prepared_UT3_v5' if i > test1_data.size else 'Prepared_UT2_v5' for i in df.index]
                save_pandas_as_artefact(df,"predict_result.csv",tmp_dir)

                # Metryki
                bacc = balanced_accuracy_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mlflow.log_metrics({"ACC": acc, "BACC": bacc, "F1": f1})
                print({"ACC": acc, "BACC": bacc, "F1": f1})

                