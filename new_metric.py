from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import mlflow

def mlflow_scoring_wrapper(y, y_pred):
    # Start new run inside GridSearchCV scoring
    # with mlflow.start_run(nested=True):  # nested to avoid clashing
    # y_pred = estimator.predict(X)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y,y_pred)
        # Log params & metrics
    # if hasattr(estimator, 'get_params'):
    #     params = estimator.get_params(deep=True)
    #     mlflow.log_params(params)
    #     try:
    #         mlflow.set_tag("prototype_n_per_class",params.get("prototype_n_per_class",None))
    #         mlflow.set_tags(params.get("solver_params"))
    #     except:pass
    # mlflow.log_metric("accuracy", acc)
    return {"acc":acc,"bacc":bacc,"f1":f1}

# def mlflow_scoring_wrapper(estimator, X, y):
#     # Start new run inside GridSearchCV scoring
#     # with mlflow.start_run(nested=True):  # nested to avoid clashing
#     y_pred = estimator.predict(X)
#     acc = accuracy_score(y, y_pred)
#     bacc = balanced_accuracy_score(y, y_pred)
#     f1 = f1_score(y,y_pred)
#         # Log params & metrics
#     # if hasattr(estimator, 'get_params'):
#     #     params = estimator.get_params(deep=True)
#     #     mlflow.log_params(params)
#     #     try:
#     #         mlflow.set_tag("prototype_n_per_class",params.get("prototype_n_per_class",None))
#     #         mlflow.set_tags(params.get("solver_params"))
#     #     except:pass
#     # mlflow.log_metric("accuracy", acc)
#     return {"acc":acc,"bacc":bacc,"f1":f1}
