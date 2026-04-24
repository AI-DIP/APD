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
        filter_string="tags.STATUS = 'ERROR'",
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
        # client.delete_tag(run_id, "WORKING")
        client.set_tag(run_id, "STATUS", "WAITING")
    except Exception as e:
        raise(e)