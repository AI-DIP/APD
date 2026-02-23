import numpy as np
from itertools import product
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
from src.adversarial_prototype_decomposition.utils.mlflow_connector import MlFlowConnector
from src.adversarial_prototype_decomposition.classifier.classifiers import APD_Classifier

class GridSearch:
    def __init__(self, estimator, param_grid: dict, scoring, n_jobs=None, cv=None, verbose: int = 0, ml_flow:MlFlowConnector=None, additional_params:dict=None) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.ml_flow=ml_flow
        self.additional_params = additional_params

        self._results = []
        self._best_params = None
        self._best_score = None
        self._best_estimator = None

    def _get_param_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for v in product(*values):
            yield dict(zip(keys, v))

    def _fit_single_fold(self, model, params, X_train, y_train, X_test, y_test, fold_number:int=None):
        if self.ml_flow is not None:
            name = None if fold_number is None else f"FOLD_{fold_number}"
            self.ml_flow.start_run(name, True, params)
            self.ml_flow.add_params(params)
            self.ml_flow.add_params(self._get_aditional_params(params))
            self.ml_flow.add_tags(self._get_aditional_params(params))
            self.ml_flow.log_dataset()
        model = clone(model)
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = self.scoring(y_test, y_pred)
        if self.ml_flow is not None:
            try:
                if type(model)==APD_Classifier:
                    acc['region_size'] = model.region_stats.shape[0]
            except:pass
            self.ml_flow.add_metrics(acc)
            self.ml_flow.stop_run()
        return acc
    
    def _get_aditional_params(self, params:dict):
        if self.additional_params is None:return
        add_params = {}
        for key in self.additional_params:
            add_params[key] = self.additional_params[key][0](params.get(self.additional_params[key][1]))
        return add_params

    def fit(self, X, y):
        for params in self._get_param_combinations():
            if(self.ml_flow is not None):
                name = ",".join(f"{n}_{params[n]}" for n in params)
                self.ml_flow.start_run(name,False,params)
                self.ml_flow.add_params(params)
                self.ml_flow.add_params(self._get_aditional_params(params))
                self.ml_flow.add_tags(self._get_aditional_params(params))
                self.ml_flow.log_dataset()

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            tasks = []


            for index, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                tasks.append((clone(self.estimator), params, X_train, y_train, X_test, y_test,index))

            if self.n_jobs != 1:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    scores = list(executor.map(lambda args: self._fit_single_fold(*args), tasks))
            else:
                scores = [self._fit_single_fold(*args) for args in tasks]
            mean_score = np.mean([d['acc'] for d in scores])
            self._results.append({'params': params, 'score': mean_score})

            if(self.ml_flow is not None):
                keys = scores[0].keys()
                avg_dict = {f"{key}-AVG": np.mean([d[key] for d in scores]) for key in keys}
                std_dict = {f"{key}-STD": np.std([d[key] for d in scores]) for key in keys}
                self.ml_flow.add_metrics(avg_dict)
                self.ml_flow.add_metrics(std_dict)
                for index, score in enumerate(scores):
                    self.ml_flow.add_metrics(score, index)
                self.ml_flow.stop_run()

            if self.verbose:
                print(f"Params: {params}, Score: {mean_score}")

            if self._best_score is None or mean_score > self._best_score:
                self._best_score = mean_score
                self._best_params = params
                self._best_estimator = clone(self.estimator).set_params(**params)
                self._best_estimator.fit(X, y)

    def predict(self, X):
        if self._best_estimator is None:
            raise Exception("Model not fitted yet. Call fit(X, y) first.")
        return self._best_estimator.predict(X)
