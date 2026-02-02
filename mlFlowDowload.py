import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime
import os
import re

MLFLOW_TRACKING_URI = "http://192.168.10.40:5000"
EXPORT_DIR = "Data"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


os.makedirs(EXPORT_DIR, exist_ok=True)

# experiments = client.list_experiments()
# experiment_name = experiment.name
# experiment_id = experiment.experiment_id

experiments = ["APD3_Prepared_UT2_v5","APD3_Prepared_UT3_v5","APD_Prepared_UT1_v5_048",
               "APD2_Prepared_UT2_v5","APD_Prepared_UT3_v5","APD_Prepared_UT1_v5_048",
               "APD_Prepared_UT2_v5","APD_Prepared_UT3_v5","APD_Prepared_UT1_v5_048",
               "DT_Prepared_UT2_v5","DT_Prepared_UT3_v5","DT_Prepared_UT1_v5_048",
               "DT2_Prepared_UT2_v5","DT2_Prepared_UT3_v5","DT2_Prepared_UT1_v5_048"]

for experiment_name in experiments:
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Nie znaleziono eksperymentu o nazwie: {experiment_name}")

    experiment_id = experiment.experiment_id
    print(f"\n Eksport eksperymentu: {experiment_name} (ID: {experiment_id})")

    runs = []
    next_token = None

    while True:
        results = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=1000,
            page_token=next_token,
        )
        runs.extend(results)
        next_token = results.token
        if not next_token:
            break

    print(f"Załadowano {len(runs)} runów")

    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0) if run.info.start_time else None,
            "end_time": datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None,
            "artifact_uri": run.info.artifact_uri,
            "experiment_id": run.info.experiment_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
        }

        for k, v in run.data.tags.items():
            row[f"t_{k}"] = v

        for k, v in run.data.params.items():
            row[f"p_{k}"] = v

        for k, v in run.data.metrics.items():
            row[f"m_{k}"] = v

        data.append(row)

    df = pd.DataFrame(data)

    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', experiment_name)
    csv_path = os.path.join(EXPORT_DIR, f"runs_{safe_name}.csv")

    df.to_csv(csv_path, index=False)
    print(f"Zapisano do pliku: {csv_path}")
