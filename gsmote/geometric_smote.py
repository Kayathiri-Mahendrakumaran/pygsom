"""Class to perform over-sampling using Geometric SMOTE."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
from imblearn.utils._validation import _count_class_sample
from numpy.linalg import norm
from sklearn.utils import check_random_state
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object, Substitution
from imblearn.utils._docstring import _random_state_docstring
import collections
from sklearn.cluster import KMeans
from gsmote.comparison_testing import compare_visual

def _make_geometric_sample(
        center, surface_point, truncation_factor, deformation_factor, random_state
):
    """A support function that returns an artificial point inside
    the geometric region defined by the center and surface points.

    Parameters
    ----------
    center : ndarray, shape (n_features, )
        Center point of the geometric region.

    surface_point : ndarray, shape (n_features, )
        Surface point of the geometric region.

    truncation_factor : float, optional (default=0.0)
        The type of truncation. The values should be in the [-1.0, 1.0] range.

    deformation_factor : float, optional (default=0.0)
        The type of geometry. The values should be in the [0.0, 1.0] range.

    random_state : int, RandomState instance or None
        Control the randomization of the algorithm.

    Returns
    -------
    point : ndarray, shape (n_features, )
            Synthetically generated sample.

    """

    # Zero radius case
    if np.array_equal(center, surface_point):
        return center

    # Generate a point on the surface of a unit hyper-sphere
    radius = norm(center - surface_point)
    normal_samples = random_state.normal(size=center.size)
    point_on_unit_sphere = normal_samples / norm(normal_samples)
    point = (random_state.uniform(size=1) ** (1 / center.size)) * point_on_unit_sphere

    # Parallel unit vector
    parallel_unit_vector = (surface_point - center) / norm(surface_point - center)

    # Truncation
    close_to_opposite_boundary = (
            truncation_factor > 0
            and np.dot(point, parallel_unit_vector) < truncation_factor - 1
    )
    close_to_boundary = (
            truncation_factor < 0
            and np.dot(point, parallel_unit_vector) > truncation_factor + 1
    )
    if close_to_opposite_boundary or close_to_boundary:
        point -= 2 * np.dot(point, parallel_unit_vector) * parallel_unit_vector

    # Deformation
    parallel_point_position = np.dot(point, parallel_unit_vector) * parallel_unit_vector
    perpendicular_point_position = point - parallel_point_position
    point = (
            parallel_point_position
            + (1 - deformation_factor) * perpendicular_point_position
    )

    # Translation
    point = center + radius * point

    return point


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class GeometricSMOTE(BaseOverSampler):
    """Class to to perform over-sampling using Geometric SMOTE.

    This algorithm is an implementation of Geometric SMOTE, a geometrically
    enhanced drop-in replacement for SMOTE as presented in [1]_.

    Read more in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    truncation_factor : float, optional (default=0.0)
        The type of truncation. The values should be in the [-1.0, 1.0] range.

    deformation_factor : float, optional (default=0.0)
        The type of geometry. The values should be in the [0.0, 1.0] range.

    selection_strategy : str, optional (default='combined')
        The type of Geometric SMOTE algorithm with the following options:
        ``'combined'``, ``'majority'``, ``'minority'``.

    k_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to use when synthetic
        samples are constructed for the minority method.  If object, an estimator
        that inherits from :class:`sklearn.neighbors.base.KNeighborsMixin` that
        will be used to find the k_neighbors.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    See the original paper: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [2]_.

    References
    ----------

    .. [1] G. Douzas, F. Bacao, "Geometric SMOTE:
       a geometrically enhanced drop-in replacement for SMOTE",
       Information Sciences, vol. 501, pp. 118-135, 2019.

    .. [2] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique", Journal of Artificial
       Intelligence Research, vol. 16, pp. 321-357, 2002.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from gsmote import GeometricSMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> gsmote = GeometricSMOTE(random_state=1)
    >>> X_res, y_res = gsmote.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})

    """

    def __init__(
            self,
            sampling_strategy='auto',
            random_state=None,
            truncation_factor=1.0,
            deformation_factor=0.0,
            # selection_strategy='combined',
            k_neighbors=5,
            n_jobs=1,
            sampling_rate=0.25,
    ):
        super(GeometricSMOTE, self).__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        # self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.sampling_rate = sampling_rate

    def _validate_estimator(self):
        """Create the necessary attributes for Geometric SMOTE."""

        # Check random state
        self.random_state_ = check_random_state(self.random_state)
        # Create nearest neighbors object for mixed class
        self.nn_mix_ = check_neighbors_object('nns_mixed', self.k_neighbors)
        # Create nearest neighbors of positive class
        self.nns_pos_ = check_neighbors_object('nns_positive', self.k_neighbors, additional_neighbor=1)
        self.nns_pos_.set_params(n_jobs=self.n_jobs)
        # Create nearest neighbors of negative class
        self.nn_neg_ = check_neighbors_object('nn_negative', nn_object=1)
        self.nn_neg_.set_params(n_jobs=self.n_jobs)

    def _make_geometric_samples(self, X, y, pos_class_label, n_samples):
        """A support function that returns an artificials samples inside
        the geometric region defined by nearest neighbors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples, )
            Corresponding label for each sample in X.
        pos_class_label : str or int
            The minority class (positive class) target value.
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        X_new : ndarray, shape (n_samples_new, n_features)
            Synthetically generated samples.
        y_new : ndarray, shape (n_samples_new, )
            Target values for synthetic samples.

        """
        # Return zero new samples
        if n_samples == 0:
            return (
                np.array([], dtype=X.dtype).reshape(0, X.shape[1]),
                np.array([], dtype=y.dtype),
            )
        # Select positive class samples
        X_pos = X[y == pos_class_label]

        # Oversampling limit for minority class
        subCluster_label, sampling_limitation = self.sub_clustering(X_pos, n_samples)

        # Get positive k_nearest points
        self.nns_pos_.fit(X_pos)
        points_pos = self.nns_pos_.kneighbors(X_pos)[1][:, 1:]
        samples_indices = self.random_state_.randint(low=0, high=len(points_pos.flatten()), size=n_samples)
        rows = np.floor_divide(samples_indices, points_pos.shape[1])
        cols = np.mod(samples_indices, points_pos.shape[1])

        # Get negative k_nearest points
        X_neg = X[y != pos_class_label]
        self.nn_neg_.fit(X_neg)
        points_neg = self.nn_neg_.kneighbors(X_pos)[1]

        # Get combined neighbours
        self.nn_mix_.fit(X)
        points_mix = self.nn_mix_.kneighbors(X_pos)[1]

       # Generate new samples
        X_new = np.zeros((n_samples, X.shape[1]))
        for ind, (row, col) in enumerate(zip(rows, cols)):
            created = False
            while not created:
                proceed = True

                # Define center point
                center = X_pos[row]

                count_min = 0
                for index in points_mix[row]:
                    if y[index] == pos_class_label:
                        count_min = count_min + 1

                # check noisy point
                if count_min > 1:

                    # Minority strategy
                    if count_min == points_mix.shape[1]:
                        # surface_point = X_pos[points_pos[row, col]]
                        remaining_limitation = sampling_limitation[subCluster_label[row]]
                        if remaining_limitation>0:
                            sampling_limitation[subCluster_label[row]] = remaining_limitation-1
                            surface_point = X_pos[points_pos[row, col]]
                        else:
                            proceed = False
                            row = self.random_state_.randint(low=0, high=len(X_pos), size=1)[0]
                    # Combined strategy
                    else:
                        surface_point_pos = X_pos[points_pos[row, col]]
                        surface_point_neg = X_neg[points_neg[row, 0]]
                        radius_pos = norm(center - surface_point_pos)
                        radius_neg = norm(center - surface_point_neg)
                        surface_point = (
                            surface_point_neg if radius_pos > radius_neg else surface_point_pos
                        )

                    # Append new sample
                    if proceed:
                        X_new[ind] = _make_geometric_sample(
                            center,
                            surface_point,
                            self.truncation_factor,
                            self.deformation_factor,
                            self.random_state_,
                        )
                        created = True
                else:
                    row=self.random_state_.randint(low=0, high=len(X_pos), size=1)[0]


        # Create new samples for target variable
        y_new = np.array([pos_class_label] * len(samples_indices))
        return X_new, y_new

    # Handles sub clustering
    def sub_clustering(self, X_pos, n_samples):
        num_clusters=4
        num_samples = n_samples * 0.5
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X_pos)
        clusters=[]
        range_oversample = []
        y_kmeans = kmeans.predict(X_pos)
        for i in range(num_clusters):
            cluster = X_pos[y_kmeans == i]
            clusters.append(cluster)
            range_oversample.append(int(len(cluster)*(num_samples/len(X_pos))))
        return y_kmeans,range_oversample

    def _fit_resample(self, X, y):
        # Validate estimator's parameters
        self._validate_estimator()
        # Copy data
        X_resampled, y_resampled = X.copy(), y.copy()
        # Resample data
        for class_label, n_samples in self.min_oversample(y).items():
            if n_samples < 0:
                n_samples = 0
            # Approach by clustering
            # clusters = self.sub_clustering(X,y,class_label)
            # Apply gsmote mechanism
            X_new, y_new = self._make_geometric_samples(X, y, class_label, n_samples)

            # for cluster in clusters:
            #     num_samples = int(n_samples*(len(cluster)/len([val for sublist in clusters for val in sublist])))
            #     X_new, y_new = self._make_geometric_samples(X, y, class_label, num_samples, cluster)
                # Append new data
            X_resampled, y_resampled = (
                np.vstack((X_resampled, X_new)),
                np.hstack((y_resampled, y_new)),
            )

        return X_resampled, y_resampled

    # find number of minority samples to generate
    def min_oversample(self, y):
        target_stats = _count_class_sample(y)
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)

        sampling_strategy = {
            key: int(n_sample_majority * (self.sampling_rate / (1 - self.sampling_rate))) - value
            for (key, value) in target_stats.items()
            if key == class_minority
        }
        sampling_strategy = collections.OrderedDict(sampling_strategy)
        return sampling_strategy
