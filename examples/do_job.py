import os
import uuid
import mlflow
import json
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from imblearn.under_sampling import ClusterCentroids
from adversarial_prototype_decomposition.sampler import GLVQ_Sampler
from adversarial_prototype_decomposition.classifier import classifiers as ppe

def get_estimator(setup: dict):
    if setup is None: return None
    model = setup.get("MODEL")
    if model == "SVM":
        return SVC(C=float(setup.get("SVM_C")), kernel=setup.get("SVM_KERNEL"),
                   gamma=0.01, cache_size=int(setup.get("SVM_CACHE_SIZE")), random_state=None)
    if model == "APD":
        baseEstimator, proto_selection, metric = None, None, 'sqeuclidean'
        if setup.get("APD_ESTIMATOR_MODEL") == "SVM":
            baseEstimator = SVC(C=float(setup.get("APD_ESTIMATOR_SVM_C")),
                                kernel=setup.get("APD_ESTIMATOR_SVM_KERNEL"),
                                gamma=0.01,
                                cache_size=int(setup.get("APD_ESTIMATOR_SVM_CACHE_SIZE")),
                                random_state=None)
        proto_type = setup.get("APD_PROTO_SELECTION")
        n_proto = int(setup.get("APD_N_PROTO"))
        if proto_type == "K-MEANS":
            proto_selection = ClusterCentroids(sampling_strategy={0: n_proto, 1: n_proto})
        elif proto_type == "GLVQ":
            proto_selection = GLVQ_Sampler(prototype_n_per_class=np.array([n_proto, n_proto]),
                                           solver_params={"step_size": 0.1, "max_runs": 100, "batch_size": 16})
        elif proto_type == "GMLVQ":
            metric = 'mahalanobis'
            proto_selection = GMLVQ_Sampler(prototype_n_per_class=np.array([n_proto, n_proto]),
                                            solver_params={"step_size": 0.1, "max_runs": 100, "batch_size": 16})
        return ppe.APD_ClassifierScaler(apd_type="apd2_middle_point", base_estimator=baseEstimator,
                                        min_support=int(setup.get("APD_MIN_SUPPORT")),
                                        unbalanced_rate=float(setup.get("APD_UNBALANCED_RATE")),
                                        metric=metric, proto_selection=proto_selection)
    return None

# ---- konfiguracja polaczenia ----
def login():
    if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "passwor"

login()
mlflow.set_tracking_uri("http://192.168.10.40:5000")

# ---- logika petli po WAITING runach ----
EXPERIMENT_NAME = "SVM_TEST3"
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError("Eksperyment nie istnieje")

experiment_id = experiment.experiment_id

while True:
    # Pobranie najstarszego runa WAITING
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.STATUS = 'WAITING'",
        max_results=1,
        order_by=["attribute.start_time ASC"]
    )

    if not runs:
        print("Brak runow WAITING. Koniec petli.")
        break

    run_id = runs[0].info.run_id
    working_id = str(uuid.uuid4())

    try:
        # Rezerwacja runa
        client.set_tag(run_id, "WORKING", working_id)
        client.set_tag(run_id, "STATUS", "RUNNING")

        # Walidacja locka
        run = client.get_run(run_id)
        if run.data.tags.get("WORKING") != working_id:
            print(f"Run {run_id} przejęty przez inny proces, pomijam.")
            continue

        # ---- praca ----
        print(f"Przetwarzanie run: {run_id}")
        tags = run.data.tags
        print("Tagi runa w trakcie pracy:")
        for k, v in tags.items():
            print(f"{k} = {v}")

        # Pobranie input datasetu
        dataset_inputs = run.inputs.dataset_inputs
        if not dataset_inputs:
            raise RuntimeError("Run nie ma zdefiniowanego input datasetu")
        dataset_input = dataset_inputs[0]
        dataset_source = dataset_input.dataset.source
        path_dict = json.loads(dataset_source)
        file_path = path_dict["uri"]

        df = pd.read_csv(file_path, sep=";", quoting=1, quotechar='"')

        # if tags.get("MODEL") == "CSVM":
        #     client.delete_tag(run_id, "WORKING")
        #     client.set_tag(run_id, "STATUS", "SKIP")
        #     continue

        X = df[[c for c in df.columns if c not in ["LABEL", "id"]]]
        y = np.squeeze(df[['LABEL']].values)
        le = LabelEncoder()
        y = le.fit_transform(y)
        ohe = OneHotEncoder(sparse_output=False)
        sym_cols = (X.dtypes == "O").values
        ct = ColumnTransformer([('ohe', ohe, sym_cols)], remainder='passthrough')
        X = ct.fit_transform(X)

        kf = StratifiedKFold(n_splits=10, shuffle=False)
        estimator = get_estimator(tags)
        mlflow.log_params(estimator.get_params(), run_id=run_id)
        cv = cross_validate(estimator, X, y, cv=kf,
                            scoring=["balanced_accuracy", 'accuracy', 'f1_macro'],n_jobs=10)

        for i in range(10):
            mlflow.log_metric("CV_ACC", cv["test_accuracy"][i], step=i, run_id=run_id)
            mlflow.log_metric("CV_BACC", cv["test_balanced_accuracy"][i], step=i, run_id=run_id)
            mlflow.log_metric("CV_F1", cv["test_f1_macro"][i], step=i, run_id=run_id)
            mlflow.log_metric("CV_Fit_Time", cv["fit_time"][i], step=i, run_id=run_id)
            mlflow.log_metric("CV_Test_Time", cv["score_time"][i], step=i, run_id=run_id)

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
        }, run_id=run_id)

        # Zwolnienie locka po zakończeniu pracy
        client.delete_tag(run_id, "WORKING")
        client.set_tag(run_id, "STATUS", "COMPLETE")
        print(f"Run {run_id} zakonczony pomyslnie.")

    except Exception as e:
        # Obsługa bledow
        error_msg = f"Wyjatek w run {run_id}: {str(e)}"
        print(error_msg)
        # zapis do pliku
        # mlflow.log_text(error_msg, f"Rules.txt")
        # zmiana statusu na ERROR
        try:
            client.delete_tag(run_id, "WORKING")
        except:
            pass
        client.set_tag(run_id, "STATUS", "ERROR")
