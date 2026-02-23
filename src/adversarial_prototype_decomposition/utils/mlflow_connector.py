import mlflow
import mlflow.data
import pandas as pd

class MlFlowConnector:
    def __init__(self, tracking_url:str, experiment_name:str=None) -> None:
        self.tracking_url = tracking_url
        self.experiment_name = experiment_name
        self.dataset_info = None
    
    def start(self):
        mlflow.set_tracking_uri(self.tracking_url)
        if mlflow.active_run():
            mlflow.end_run()
        if self.experiment_name is not None:
            mlflow.set_experiment(self.experiment_name)
    
    def rename_experiment(self, experiment_name:str):
        self.experiment_name = experiment_name
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self,name:str = None, nested:bool=False, tags:dict = None):
        if mlflow.active_run() and not nested:
            mlflow.end_run()
        mlflow.start_run(
            run_name = name,
            nested = nested,
        )
        for tag in tags:
            mlflow.set_tag(tag,tags[tag])
    
    def stop_run(self):
        if mlflow.active_run():
            mlflow.end_run()
    
    def add_metrics(self, metrics:dict, step:int=False):
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
    
    def add_metric(self, metric_name:str, metric_value, step:int=False):
        if mlflow.active_run():
            mlflow.log_metric(metric_name, metric_value,step=step)
    
    def add_params(self, params:dict):
        if params is None:return
        if mlflow.active_run():
            mlflow.log_params(params)
    
    def add_param(self, param_name:str, param_value):
        if mlflow.active_run():
            mlflow.log_param(param_name, param_value)
    
    def add_tags(self, params:dict):
        if params is None:return
        if mlflow.active_run():
            mlflow.set_tags(params)
    
    def add_tag(self, param_name:str, param_value):
        if mlflow.active_run():
            mlflow.set_tag(param_name, param_value)
    
    def add_dataset(self,dataset:pd.DataFrame, dataset_path:str, dataset_name:str):
        self.dataset_info = mlflow.data.from_pandas(dataset, source=dataset_path, name=dataset_name)
    
    def log_dataset(self):
        if self.dataset_info is None:return
        if mlflow.active_run():
            mlflow.log_input(self.dataset_info)
            
