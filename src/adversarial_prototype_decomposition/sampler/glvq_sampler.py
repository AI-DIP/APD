"""Class to perform under-sampling by generating centroids based on
clustering."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
from sklvq import GLVQ, GMLVQ
from imblearn.under_sampling.base import BaseUnderSampler
from imblearn.under_sampling import ClusterCentroids
from sklearn.utils import check_random_state

VOTING_KIND = ("auto", "hard", "soft")


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class GLVQ_Sampler(BaseUnderSampler):
    """Undersample by generating centroids based on GLVQ method.

    Method that under samples the dataset by replacing the input data
    by the prototypes of a GLVQ algorithm.  This algorithm applys
    GLVQ to entire dataset, and sample labels are obtained by prototype
    labels returned by the GLVQ model.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    estimator : estimator object, default=None
        A scikit-learn compatible clustering method that exposes a `n_clusters`
        parameter and a `cluster_centers_` fitted attribute. By default, it will
        be a default :class:`~sklearn.cluster.KMeans` estimator.

    n_clusters: the number of cluster centers


    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated estimator created from the `estimator` parameter.

    voting_ : str
        The validated voting strategy.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    EditedNearestNeighbours : Under-sampling by editing samples.

    CondensedNearestNeighbour: Under-sampling by condensing samples.

    ClusterCentroids: Under-sampling by kmeans of the majority classes

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> from imblearn.under_sampling import ClusterCentroids
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> cc = ClusterCentroids(
    ...     estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
    ... )
    >>> X_res, y_res = cc.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{...}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "random_state": ["random_state"],
        "prototype_n_per_class":"no_validation",
        "solver_params":"no_validation"
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        prototype_n_per_class = None,
        solver_params = None,
        random:str = None
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.prototype_n_per_class = prototype_n_per_class
        self.solver_params = solver_params
        self.estimator_ = None
        self.random = random

    def _fit_resample(self, X, y):
        self.estimator_ = GLVQ(activation_type="swish",
                           solver_type="sgd",
                           prototype_n_per_class=self.prototype_n_per_class,
                           random_state=self.random_state,
                           solver_params=self.solver_params, )
        if self.random is None:
            self.estimator_.fit(X,y)
        elif self.random == "kmeans":
            kmeans = ClusterCentroids(sampling_strategy={0: self.prototype_n_per_class[0], 1: self.prototype_n_per_class[1]})
            Xp,yp = kmeans.fit_resample(X,y)
            X,y = self.estimator_._check_data_and_labels(X,y)
            self.estimator_.random_state_ = check_random_state(self.estimator_.random_state)
            self.estimator_._before_fit(X,y)
            self.estimator_.set_prototypes(Xp)
            self.estimator_._solver.solve(X, y, self.estimator_)
            self.estimator_._after_fit(X, y)
        X_resampled = np.copy(self.estimator_.prototypes_)
        y_resampled = np.copy(self.estimator_.prototypes_labels_)
        return X_resampled, y_resampled

    def get_lambda(self):
        return self.estimator_.lambda_

    def _more_tags(self):
        return {"sample_indices": False}

class GMLVQ_Sampler(BaseUnderSampler):
    """Undersample by generating centroids based on GLVQ method.

    Method that under samples the dataset by replacing the input data
    by the prototypes of a GLVQ algorithm.  This algorithm applys
    GLVQ to entire dataset, and sample labels are obtained by prototype
    labels returned by the GLVQ model.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    estimator : estimator object, default=None
        A scikit-learn compatible clustering method that exposes a `n_clusters`
        parameter and a `cluster_centers_` fitted attribute. By default, it will
        be a default :class:`~sklearn.cluster.KMeans` estimator.

    n_clusters: the number of cluster centers


    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    estimator_ : estimator object
        The validated estimator created from the `estimator` parameter.

    voting_ : str
        The validated voting strategy.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    EditedNearestNeighbours : Under-sampling by editing samples.

    CondensedNearestNeighbour: Under-sampling by condensing samples.

    ClusterCentroids: Under-sampling by kmeans of the majority classes

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> from imblearn.under_sampling import ClusterCentroids
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> cc = ClusterCentroids(
    ...     estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42
    ... )
    >>> X_res, y_res = cc.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{...}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "random_state": ["random_state"],
        "prototype_n_per_class":"no_validation",
        "solver_params":"no_validation"
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        prototype_n_per_class = None,
        solver_params = None,
        random:str = None
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.prototype_n_per_class = prototype_n_per_class
        self.solver_params = solver_params
        self.estimator_ = None
        self.random = random

    def _fit_resample(self, X, y):
        self.estimator_ = GMLVQ(activation_type="swish",
                           solver_type="sgd",
                           prototype_n_per_class=self.prototype_n_per_class,
                           random_state=self.random_state,
                           solver_params=self.solver_params, )
        if self.random is None:
            self.estimator_.fit(X,y)
        elif self.random == "kmeans":
            X,y = self.estimator_._check_data_and_labels(X,y)
            self.estimator_.random_state_ = check_random_state(self.estimator_.random_state)
            self.estimator_._before_fit(X,y)
            kmeans = ClusterCentroids(sampling_strategy={0: self.prototype_n_per_class[0], 1: self.prototype_n_per_class[1]})
            Xp,yp = kmeans.fit_resample(X,y)
            self.estimator_.set_prototypes(Xp)
            self.estimator_._solver.solve(X, y, self.estimator_)
            self.estimator_._after_fit(X, y)
        X_resampled = np.copy(self.estimator_.prototypes_)
        y_resampled = np.copy(self.estimator_.prototypes_labels_)
        return X_resampled, y_resampled

    def get_lambda(self):
        return self.estimator_.lambda_

    def _more_tags(self):
        return {"sample_indices": False}