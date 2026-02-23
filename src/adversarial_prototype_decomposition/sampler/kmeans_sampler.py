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
from imblearn.utils._param_validation import HasMethods, StrOptions
from imblearn.under_sampling.base import BaseUnderSampler

VOTING_KIND = ("auto", "hard", "soft")


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class SimpleClusterCentroids(BaseUnderSampler):
    """Undersample by generating centroids based on clustering methods.

    Method that under samples the dataset by replacing the input data
    by the cluster centroid of a KMeans algorithm.  This algorithm applys
    kmeans to entire dataset, and sample labels are obtained by identifining
    the most frequent class within the cluster.
    Infact this simply applies kmeans and additionally id generates y by taking
    the most frequent sample

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
        "estimator": [HasMethods(["fit", "predict"]), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        estimator=None,
        n_clusters = 10,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.estimator = estimator
        self.n_clusters = n_clusters

    def _validate_estimator(self):
        """Private function to create the KMeans estimator"""
        if self.estimator is None:
            self.estimator_ = KMeans(random_state=self.random_state, n_clusters=self.n_clusters, n_init="auto")
        else:
            self.estimator_ = clone(self.estimator)
            if "n_clusters" not in self.estimator_.get_params():
                raise ValueError(
                    "`estimator` should be a clustering estimator exposing a parameter"
                    " `n_clusters` and a fitted parameter `cluster_centers_`."
                )

    def _generate_sample(self, X, y, centroids):
        clust_res = self.estimator_.predict(X)
        n = self.estimator_.cluster_centers_.shape[0]
        lab_encoder = LabelEncoder()
        y_p = lab_encoder.fit_transform(y)
        y_new = -np.ones((n,),dtype=y_p.dtype)
        for i in range(n):
            counts = np.bincount(y_p[clust_res==i])
            id_max = np.argmax(counts)
            y_new[i] = id_max
        y_new = lab_encoder.inverse_transform(y_new)

        if sparse.issparse(X):
            X_new = sparse.csr_matrix(centroids, dtype=X.dtype)
        else:
            X_new = centroids
        return X_new, y_new

    def _fit_resample(self, X, y):
        self._validate_estimator()
        self.estimator_.set_params(**{"n_clusters": self.n_clusters})
        self.estimator_.fit(X)
        if not hasattr(self.estimator_, "cluster_centers_"):
            raise RuntimeError(
                "`estimator` should be a clustering estimator exposing a "
                "fitted parameter `cluster_centers_`."
            )
        X_resampled, y_resampled = self._generate_sample(
            X,y,self.estimator_.cluster_centers_)

        return X_resampled, np.array(y_resampled, dtype=y.dtype)

    def _more_tags(self):
        return {"sample_indices": False}