import numpy as np
import pandas as pd
import sklearn
import pickle

from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
from imblearn.base import SamplerMixin
from adversarial_prototype_decomposition.base.apd import APD, APD2, APD3, PE, APD2_MIDDLE_POINT, APD2_MULTI_CLASS, APD_MULTI_CLASS
from joblib import Parallel, delayed
import copy
from scipy.spatial.distance import cdist
from sklearn.dummy import DummyClassifier

class APD_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 base_estimator=RandomForestClassifier(),
                 apd_type="apd",
                 unbalanced_rate=0.3,
                 min_support=500,
                 minimum_regions=1,
                 proto_selection={0: 10, 1: 10},
                 prune_regions=True,
                 n_jobs=None,
                 metric:str =  'sqeuclidean'
                 ):
        """
        Constructor for the APD_ensemble class.
        The idea of this algorithm is presented in (to appear)
        Basically it splits the datasets into smaller once using a distance to the nearest reference prototypes from
        to separate classes
        :param base_estimator: the base estimator to use for fitting the within each regino
        :param unbalanced_rate: if a region has unbalanced number of samples from the two classes the the region will be
         merged with another region that exist. Here we calculate it as min(c1/c2,c2/c1) so it shows the ration of the minority to majority class within region
        :param min_support: minimum number of samples per region
        :param proto_selection: a protothpe selectin method. Possible options are:
            dict wher kays are class label names and values represent the number of prototypes which will be randomly
            selected from samples of given class
            an algorithm from imblearn packated or some other algorithm which supports fit_resample method
        """
        self.base_estimator = base_estimator
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.proto_selection = proto_selection
        self.apd_type = apd_type
        self.minimum_regions = minimum_regions = 2
        self.prune_regions = prune_regions
        self.n_jobs = n_jobs
        self.metric = metric

    def _initialize_apd(self, X, y):
        if type(self.proto_selection) == dict:
            idx_all = np.zeros((y.shape[0]), dtype=bool)
            for label, n_samples in self.proto_selection.items():
                idClass = np.nonzero(y == label)[0]
                idx = sklearn.utils.resample(np.arange(idClass.shape[0]), n_samples=n_samples, replace=False)
                idx_all[idClass[idx]] = True
            Xp = X[idx_all, :]  # X of selected prototypes
            yp = y[idx_all]  # Y of selected prototypes
        elif issubclass(type(self.proto_selection), tuple):
            Xp, yp = self.proto_selection
        elif issubclass(type(self.proto_selection), SamplerMixin):
            Xp, yp = self.proto_selection.fit_resample(X, y)
        else:
            raise ValueError("Unknown prototype selection method")
        if self.apd_type == "apd":
            apd = APD(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                      minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                      metric=self.metric)
        elif self.apd_type == "apd2":
            apd = APD2(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        elif self.apd_type == "apd2_middle_point":
            apd = APD2_MIDDLE_POINT(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        elif self.apd_type == "apd3":
            apd = APD3(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        elif self.apd_type == "pe":
            apd = PE(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support, prune_regions=True,
                     minimum_n_regions=self.minimum_regions,metric=self.metric)
        else:
            raise ValueError("Unknown APD/PE type. Only (apd,apd2,apd3,pe) are avaliable")

        return apd

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        """
        Train and algorithm, it first starts be identifing region and then for each region it trains the base model
        The trained models are stored in self.fitted_base_models_ attribute which is a dict where keys are prototype
        pairs (see Cantar pairing function, or APD class) and values are traind models
        :param X: training samples delivered as numpy arrays or a DataFrame
        :param y: training sample labels delivered as numpy arrays or a DataFrame
        :return: self - trained model
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        apd = self._initialize_apd(X, y)
        self.proto_ensemble_ = apd

        regions = apd.generate_regions(X, y)
        # pairs = apd.assign_regions(X,regions)
        # _ux_regions, ux_regions_counts = np.unique(pairs, return_counts=True)
        # assert np.all(np.sort(ux_regions) == np.sort(_ux_regions))
        self.regions_ = list(regions.keys())
        self.region_stats = apd.region_stats
        self.fitted_base_models_ = {}
        modelsInputData = []
        for region in regions:
            id = regions[region]
            if np.sum(id) == 0: continue
            Xm = X[id, :]
            ym = y[id]
            model = copy.deepcopy(self.base_estimator)
            modelsInputData.append((region, Xm, ym, model))

        if self.n_jobs is not None:
            parrTrainFun = lambda region, Xm, ym, model: (region, model.fit(Xm, ym))
            with Parallel(n_jobs=self.n_jobs) as parallel:
                res_all = parallel(delayed(parrTrainFun)(*input) for input in modelsInputData)
                self.fitted_base_models_ = {region: model for region, model in res_all}
        else:
            self.fitted_base_models_ = {region: model.fit(Xm, ym) for region, Xm, ym, model in modelsInputData}
        return self

    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Method used for predicting the output of the model. For each sample in X it determines the nearest region out of
         the existing region. And then based on the index of existing region it takes the classifier and performs prediction
        :param X: samples to be classified
        :return: predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        regions_ = self.regions_
        sample2region = self.proto_ensemble_.assign_regions(X, regions_)  # For each sample in X get its nearest region
        yp = np.zeros(X.shape[0], dtype=int)  # Allocate memory
        for region in regions_:  # Iterate over reginos
            id = sample2region[region]  # Get samples which belong to region pair
            Xm = X[id, :]
            if Xm.size:
                model = self.fitted_base_models_[region]  # Take the classifier associated to region "pair"
                yp[id] = model.predict(Xm)  # Make prediction using the classifier assigned to region "pair"
        return yp
    
    def predict_proba(self, X:np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        
        #Tensor sizes
        N = X.shape[0]
        C = len(self.classes_)
        K = len(self.regions_)
        proba = np.zeros((N,C,K))
        
        #Common index for all classes
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        
        for k, region in enumerate(self.regions_):
            est = self.fitted_base_models_[region]
            region_proba = est.predict_proba(X)
            for local_idx, cls in enumerate(est.classes_):
                global_idx = class_to_idx[cls]
                proba[:, global_idx, k] = region_proba[:, local_idx]
        return proba
    
    def get_competency(self, X:np.ndarray, tau:float = 1.0):
        #Validation
        check_is_fitted(self)
        X = check_array(X)
        #Get Distances
        regions_ = self.regions_
        regions_dist = self.proto_ensemble_.get_distance_to_region_prototypes(X, regions_)
        #Softmax
        scores = -regions_dist/tau
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        competency = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return competency


class APD_Multi_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 proto_selection,
                 base_estimator=RandomForestClassifier(),
                 apd_type="apd2_multi_class",
                 unbalanced_rate=0.3,
                 min_support=500,
                 minimum_regions=1,
                 prune_regions=True,
                 n_jobs=None,
                 metric:str =  'sqeuclidean',
                 ):
        self.proto_selection = proto_selection
        self.base_estimator = base_estimator
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.apd_type = apd_type
        self.minimum_regions = minimum_regions = 2
        self.prune_regions = prune_regions
        self.n_jobs = n_jobs
        self.metric = metric
    
    def _initialize_apd(self, X, y)->APD2_MULTI_CLASS:
        if issubclass(type(self.proto_selection), tuple):
            Xp, yp = self.proto_selection
        elif issubclass(type(self.proto_selection), SamplerMixin):
            Xp, yp = self.proto_selection.fit_resample(X, y)
        else:
            raise ValueError("Unknown prototype selection method")
        if self.apd_type == "apd2_multi_class":
            apd = APD2_MULTI_CLASS(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        elif self.apd_type == "apd_multi_class":
            apd = APD_MULTI_CLASS(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        else:
            raise ValueError("Unknown APD/PE type. Only (apd2_multi_class) are avaliable")

        return apd

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        """
        Train and algorithm, it first starts be identifing region and then for each region it trains the base model
        The trained models are stored in self.fitted_base_models_ attribute which is a dict where keys are prototype
        pairs (see Cantar pairing function, or APD class) and values are traind models
        :param X: training samples delivered as numpy arrays or a DataFrame
        :param y: training sample labels delivered as numpy arrays or a DataFrame
        :return: self - trained model
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        apd = self._initialize_apd(X, y)
        self.proto_ensemble_ = apd
        regions = apd.generate_regions(X, y)
        # pairs = apd.assign_regions(X,regions)
        # _ux_regions, ux_regions_counts = np.unique(pairs, return_counts=True)
        # assert np.all(np.sort(ux_regions) == np.sort(_ux_regions))
        self.regions_ = list(regions.keys())
        self.region_stats = apd.region_stats
        self.fitted_base_models_ = {}
        modelsInputData = []
        for region in regions:
            id = regions[region]
            model = copy.deepcopy(self.base_estimator)
            if np.sum(id) == 0: 
                model = DummyClassifier(strategy="constant", constant=0)
            Xm = X[id, :]
            ym = y[id]
            modelsInputData.append((region, Xm, ym, model))


        if self.n_jobs is not None:
            parrTrainFun = lambda region, Xm, ym, model: (region, model.fit(Xm, ym))
            with Parallel(n_jobs=self.n_jobs) as parallel:
                res_all = parallel(delayed(parrTrainFun)(*input) for input in modelsInputData)
                self.fitted_base_models_ = {region: model for region, model in res_all}
        else:
            self.fitted_base_models_ = {
                region: (model.fit([[0]], [0]) if len(Xm) == 0 else model.fit(Xm, ym))
    for region, Xm, ym, model in modelsInputData
}
        return self
    
    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Method used for predicting the output of the model. For each sample in X it determines the nearest region out of
         the existing region. And then based on the index of existing region it takes the classifier and performs prediction
        :param X: samples to be classified
        :return: predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        regions_ = self.regions_
        print(self.fitted_base_models_)
        sample2region = self.proto_ensemble_.assign_regions(X, regions_)  # For each sample in X get its nearest region
        yp = np.zeros(X.shape[0], dtype=int)  # Allocate memory
        for region in regions_:  # Iterate over reginos
            id = sample2region[region]  # Get samples which belong to region pair
            Xm = X[id, :]
            if Xm.size:
                model = self.fitted_base_models_[region]  # Take the classifier associated to region "pair"
                yp[id] = model.predict(Xm)  # Make prediction using the classifier assigned to region "pair"
        return yp

    def get_protos(self):
        check_is_fitted(self)
        return (self.proto_ensemble_.proto,self.proto_ensemble_.proto_labels)

class APD_Multi_Classifier_Proto_Train(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 proto_selection,
                 base_estimator=RandomForestClassifier(),
                 apd_type="apd2_multi_class",
                 unbalanced_rate=0.3,
                 min_support=500,
                 minimum_regions=1,
                 prune_regions=True,
                 n_jobs=None,
                 metric:str =  'sqeuclidean',
                 ):
        self.proto_selection = proto_selection
        self.base_estimator = base_estimator
        self.unbalanced_rate = unbalanced_rate
        self.min_support = min_support
        self.apd_type = apd_type
        self.minimum_regions = minimum_regions = 2
        self.prune_regions = prune_regions
        self.n_jobs = n_jobs
        self.metric = metric
    
    def _initialize_apd(self, X, y)->APD2_MULTI_CLASS:
        if issubclass(type(self.proto_selection), tuple):
            Xp, yp = self.proto_selection
        elif issubclass(type(self.proto_selection), SamplerMixin):
            Xp, yp = self.proto_selection.fit_resample(X, y)
        else:
            raise ValueError("Unknown prototype selection method")
        if self.apd_type == "apd2_multi_class":
            apd = APD2_MULTI_CLASS(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        elif self.apd_type == "apd_multi_class":
            apd = APD_MULTI_CLASS(Xp, yp, unbalanced_rate=self.unbalanced_rate, min_support=self.min_support,
                       minimum_n_regions=self.minimum_regions, prune_regions=self.prune_regions,
                       metric=self.metric)
        else:
            raise ValueError("Unknown APD/PE type. Only (apd2_multi_class) are avaliable")

        return apd

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        """
        Train and algorithm, it first starts be identifing region and then for each region it trains the base model
        The trained models are stored in self.fitted_base_models_ attribute which is a dict where keys are prototype
        pairs (see Cantar pairing function, or APD class) and values are traind models
        :param X: training samples delivered as numpy arrays or a DataFrame
        :param y: training sample labels delivered as numpy arrays or a DataFrame
        :return: self - trained model
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        apd = self._initialize_apd(X, y)
        self.proto_ensemble_ = apd
        regions = apd.generate_regions(X, y)
        # pairs = apd.assign_regions(X,regions)
        # _ux_regions, ux_regions_counts = np.unique(pairs, return_counts=True)
        # assert np.all(np.sort(ux_regions) == np.sort(_ux_regions))
        self.regions_ = list(regions.keys())
        self.region_stats = apd.region_stats
        self.fitted_base_models_ = {}
        modelsInputData = []
        
        proto_ids = set()
        for region in regions:
            p0, p1 = apd.unpairCantor(region)
            proto_ids.add(p0)
            proto_ids.add(p1)

        proto_ids = list(proto_ids)

        # budujemy macierz P
        P = np.array([apd.proto[p] for p in proto_ids])

        # odległości X → P
        distances = cdist(X, P, metric='sqeuclidean')

        # najbliższy proto dla każdego X
        closest = np.argmin(distances, axis=1)

        used_protos = {}

        for i, pid in enumerate(proto_ids):
            used_protos[pid] = {
                "CORDS": apd.proto[pid],
                "mask": closest == i
            }
        
        for region in regions:
            p0, p1 = apd.unpairCantor(region)
            
            id = used_protos[p0]["mask"] | used_protos[p1]["mask"]
            model = copy.deepcopy(self.base_estimator)
            if np.sum(id) == 0: 
                model = DummyClassifier(strategy="constant", constant=0)
            Xm = X[id, :]
            ym = y[id]
            modelsInputData.append((region, Xm, ym, model))


        if self.n_jobs is not None:
            parrTrainFun = lambda region, Xm, ym, model: (region, model.fit(Xm, ym))
            with Parallel(n_jobs=self.n_jobs) as parallel:
                res_all = parallel(delayed(parrTrainFun)(*input) for input in modelsInputData)
                self.fitted_base_models_ = {region: model for region, model in res_all}
        else:
            self.fitted_base_models_ = {
                region: (model.fit([[0]], [0]) if len(Xm) == 0 else model.fit(Xm, ym))
    for region, Xm, ym, model in modelsInputData
}
        return self
    
    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Method used for predicting the output of the model. For each sample in X it determines the nearest region out of
         the existing region. And then based on the index of existing region it takes the classifier and performs prediction
        :param X: samples to be classified
        :return: predicted labels
        """
        check_is_fitted(self)
        X = check_array(X)
        regions_ = self.regions_
        print(self.fitted_base_models_)
        sample2region = self.proto_ensemble_.assign_regions(X, regions_)  # For each sample in X get its nearest region
        yp = np.zeros(X.shape[0], dtype=int)  # Allocate memory
        for region in regions_:  # Iterate over reginos
            id = sample2region[region]  # Get samples which belong to region pair
            Xm = X[id, :]
            if Xm.size:
                model = self.fitted_base_models_[region]  # Take the classifier associated to region "pair"
                yp[id] = model.predict(Xm)  # Make prediction using the classifier assigned to region "pair"
        return yp

    def get_protos(self):
        check_is_fitted(self)
        return (self.proto_ensemble_.proto,self.proto_ensemble_.proto_labels)

class EPPE_Classifier(VotingClassifier):
    def __init__(self,
                 apd_estimator: APD_Classifier,
                 n_estimators: int = 10,
                 voting="hard",
                 weights=None,
                 n_jobs=None,
                 flatten_transform=True,
                 verbose=False,
                 ):
        self.apd_estimator = apd_estimator
        self.n_estimators = n_estimators
        estimators = [(f"APD_{i}", clone(apd_estimator)) for i in range(n_estimators)]
        super().__init__(estimators=estimators,
                         voting=voting,
                         weights=weights,
                         n_jobs=n_jobs,
                         flatten_transform=flatten_transform,
                         verbose=verbose)

class APD_ClassifierScaler(APD_Classifier):
    proto_scaled = False
    def __init__(self, 
                 base_estimator=RandomForestClassifier(), 
                 apd_type="apd", 
                 unbalanced_rate=0.3, 
                 min_support=500, 
                 minimum_regions=1, 
                 proto_selection={ 0: 10,1: 10 }, 
                 prune_regions=True, 
                 n_jobs=None, 
                 metric: str = 'sqeuclidean',
                 scaler:StandardScaler = StandardScaler(),
                 scaler_mode:str = 'all'):
        
        self.scaler = scaler
        if isinstance(self.scaler, (bytes, bytearray)):
            try:
                self.scaler_ = pickle.loads(self.scaler)
            except Exception as e:
                pass
        else:
            self.scaler_ = scaler
        self.scaler_mode = scaler_mode
        """
        scaler_mode:str = all(Prototypes/Estimator)/ est(Estimator)/ proto(Prototypes)
        """
        super().__init__(base_estimator, apd_type, unbalanced_rate, min_support, minimum_regions, proto_selection, prune_regions, n_jobs, metric)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        """
        Train and algorithm, it first starts be identifing region and then for each region it trains the base model
        The trained models are stored in self.fitted_base_models_ attribute which is a dict where keys are prototype
        pairs (see Cantar pairing function, or APD class) and values are traind models
        :param X: training samples delivered as numpy arrays or a DataFrame
        :param y: training sample labels delivered as numpy arrays or a DataFrame
        :return: self - trained model
        """
        X, y = check_X_y(X, y)
        if self.scaler_mode in ["all", "proto"]:
            X = self.scaler_.fit_transform(X)
        self.classes_ = unique_labels(y)

        apd = self._initialize_apd(X, y)
        self.proto_ensemble_ = apd

        regions = apd.generate_regions(X, y)
        # pairs = apd.assign_regions(X,regions)
        # _ux_regions, ux_regions_counts = np.unique(pairs, return_counts=True)
        # assert np.all(np.sort(ux_regions) == np.sort(_ux_regions))
        self.regions_ = list(regions.keys())
        self.region_stats = apd.region_stats
        self.fitted_base_models_ = {}
        modelsInputData = []
        for region in regions:
            id = regions[region]
            if np.sum(id) == 0: continue
            Xm = X[id, :]
            if self.scaler_mode in ["all", "est"]:
                Xm = self.scaler_.inverse_transform(Xm)
            ym = y[id]
            model = copy.deepcopy(self.base_estimator)
            modelsInputData.append((region, Xm, ym, model))

        if self.n_jobs is not None:
            parrTrainFun = lambda region, Xm, ym, model: (region, model.fit(Xm, ym))
            with Parallel(n_jobs=self.n_jobs) as parallel:
                res_all = parallel(delayed(parrTrainFun)(*input) for input in modelsInputData)
                self.fitted_base_models_ = {region: model for region, model in res_all}
        else:
            self.fitted_base_models_ = {region: model.fit(Xm, ym) for region, Xm, ym, model in modelsInputData}
        return self

    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Method used for predicting the output of the model. For each sample in X it determines the nearest region out of
         the existing region. And then based on the index of existing region it takes the classifier and performs prediction
        :param X: samples to be classified
        :return: predicted labels
        """
        print("AAAAA")
        check_is_fitted(self)
        X = check_array(X)
        if self.scaler_mode in ["all", "proto"]:
            X = self.scaler_.transform(X)
        regions_ = self.regions_
        sample2region = self.proto_ensemble_.assign_regions(X, regions_)  # For each sample in X get its nearest region
        yp = np.zeros(X.shape[0], dtype=int)  # Allocate memory
        for region in regions_:  # Iterate over reginos
            id = sample2region[region]  # Get samples which belong to region pair
            Xm = X[id, :]
            if len(Xm)>0:
                if self.scaler_mode in ["all", "est"]:
                    Xm = self.scaler_.inverse_transform(Xm)
                model = self.fitted_base_models_[region]  # Take the classifier associated to region "pair"
                yp[id] = model.predict(Xm)  # Make prediction using the classifier assigned to region "pair"
        return yp
    
class RandomOracle(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, min_size):
        self.base_estimator = base_estimator
        self.min_size = min_size

    def _splitData(self, X):
        d = cdist(X, self.proto_, 'sqeuclidean')
        id = np.argmin(d, axis=1)
        return id == 0,id == 1

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        X, y = check_X_y(X, y)
        chk = True
        while chk:
            chk = True
            idx = np.random.randint(X.shape[0], size=2)
            proto = X[idx, :].copy()
            self.proto_ = proto
            id1,id2 = self._splitData(X)
            ux1,co_ux1 = np.unique(y[id1],return_counts=True)
            ux2, co_ux2 = np.unique(y[id2], return_counts=True)
            cl = len(self.classes_)
            c1 = len(ux1)
            c2 = len(ux2)
            if (cl==c1) and (c2==cl):
                chk = False
                for co1,co2 in zip(co_ux1,co_ux2):
                    if (co1<self.min_size) or (co2<self.min_size):
                        chk = True
                        continue


        X1 = X[id1,:]
        y1 = y[id1]
        X2 = X[id2,:]
        y2 = y[id2]
        self.model1_ = clone(self.base_estimator)
        self.model2_ = clone(self.base_estimator)
        self.model1_.fit(X1,y1)
        self.model2_.fit(X2,y2)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        id1,id2 = self._splitData(X)
        X1 = X[id1, :]
        X2 = X[id2, :]
        yp1 = self.model1_.predict(X1)
        yp2 = self.model2_.predict(X2)
        yp = np.zeros((X.shape[0],),dtype=int)
        yp[id1] = yp1
        yp[id2] = yp2
        return yp

class RandomOracle_Classifier(VotingClassifier):
    def __init__(self,
                 base_estimator: RandomOracle(base_estimator=RandomForestClassifier(),min_size=100),
                 n_estimators: int = 30,
                 voting="hard",
                 weights=None,
                 n_jobs=None,
                 flatten_transform=True,
                 verbose=False,
                 ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        estimators = [(f"RandomOracle_{i}", clone(base_estimator)) for i in range(n_estimators)]
        super().__init__(estimators,
                         voting=voting,
                         weights=weights,
                         n_jobs=n_jobs,
                         flatten_transform=flatten_transform,
                         verbose=verbose)




