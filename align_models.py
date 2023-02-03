"""
This module provides utility functions for aligning Markov states between
different training attempts (different models trained for a same system)
and different systems (aligning the ensembles of models for different systems).
"""
import numpy as np
import ot
import argparse
from typing import Optional, Tuple, List
from scipy.optimize import linear_sum_assignment
from src.utils import update_yaml
from config.paths import LOCAL_SORTERS_PATH_TEMPLATE, CLUSTER_AVG_PATH_TEMPLATE, SYSTEM_SORTERS_PATH_TEMPLATE
from config.data_model_params import NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, REFERENCE_SYSTEM_FOR_ALIGNMENT

class AlignKMeans:
    def __init__(self, n_clusters: int, max_size: int) -> None:
        """
        K-Means clustering for the alignment of Markov states between different
        training attempts.

        Each estimated cluster obeys the constraint that it must contain
        exactly one point from each training attempt.

        Parameters
        ----------
        n_clusters: int
            Number of cluster, which corresponds to the number of Markov states.
        max_size: int
            Maximum size of each cluster, which corresponds to the number of
            training attempts.

        Attributes
        ----------
        centres: np.ndarray (n_clusters, D)
            Centres of the clusters. D = dimension of the features.
        labels: np.ndarray (max_size, n_clusters)
            Labels assigned to the data points.
        """
        self.n_clusters = n_clusters
        self.max_size = max_size

    def init_centres(self, X: np.ndarray) -> None:
        """
        Initialise the cluster centres by choosing random points from the data.

        Parameters
        ----------
        X: np.ndarray (max_size, n_clusters, D)
            Data in the following format: D = feature dimension.
        """
        X_flat = X.reshape(-1, X.shape[-1])
        selected_observations = np.random.choice(len(X_flat),
                                                 size=self.n_clusters,
                                                 replace=False)
        self.centres = X_flat[selected_observations]

    def step(self, X: np.ndarray) -> None:
        """
        Do a K-Means step. Cluster labels are assigned by closest proximity
        (under the alignment constraint) then centres are updated by the mean
        of each cluster.

        Parameters
        ----------
        X: np.ndarray (max_size, n_clusters, D)
            Data in the following format: D = feature dimension.
        """
        # Assign labels
        self.labels = - np.ones(X.shape[:-1], dtype=int)
        U = np.broadcast_to(X, (self.n_clusters, *X.shape)) \
            - self.centres[:, np.newaxis, np.newaxis, :]
        distances = np.linalg.norm(U, axis=-1)
        sorted_args = list(zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)))
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)
        for label, attempt, macrostate in sorted_args:
            if self.labels[attempt, macrostate] == -1 \
                    and label not in self.labels[attempt] \
                    and cluster_sizes[label] < self.max_size:
                self.labels[attempt, macrostate] = label
                cluster_sizes[label] += 1

        # Update centres
        for label in range(self.n_clusters):
            self.centres[label] = X[self.labels == label].mean(axis=0)

    def fit(self, X: np.ndarray, eps: float = 1e-3) -> None:
        """
        Execute the K-Means algorithm on the given data.

        Parameters
        ----------
        X: np.ndarray (A, n, D)
            Data in the following format: D = feature dimension.
        eps: float
            Precision threshold for stopping the algorithm.
        """
        self.init_centres(X)

        while True:
            old_centres = self.centres.copy()
            self.step(X)

            if np.linalg.norm(self.centres - old_centres) <= eps:
                break

def local_sort_mindist_kmeans(num_markov_states: int, num_models: int, avg_cluster_mindists: np.ndarray) -> np.ndarray:
    """
    Alignment using K-Means on the cluster averaged inter-residue heavy
    atom minimum distance matrices.

    Parameters
    ----------
    num_markov_states, int - number of Markov states for the underlying models
    num_models - number of models in the model ensemble for a given dataset
    avg_cluster_mindists, np.ndarray - shape = (num_models, num_markov_states, num_features)
                        precomputed avg features (interresidue distances) for the clusters as estimated by the models

    Returns np.ndarray - shape = (num_models, num_markov_states)
    -------
    """
    clustering = AlignKMeans(num_markov_states, num_models)
    clustering.fit(avg_cluster_mindists)
    sorter = np.argsort(clustering.labels, axis=1)
    return sorter

def best_sorter(C: np.ndarray, threshold: Optional[float] = None,
                maximize: bool = False) -> Tuple[List[int]]:
    """
    Find the best alignment sorter from a cost (resp. similarity) matrix C, using
    the Hungarian algorithm (implemented in the package SciPy).

    The outputted sorter ``best_sorter`` is such that ``C[:, best_sorter]`` has
    minimal (resp. maximal) trace.

    A threshold value may also be inputted, which allow to ignore cost
    (resp. similarity) values that are above (resp. below) the threshold.

    Parameters
    ----------
    C: np.ndarray (n, n)
        Cost or similarity matrix. n = number of Markov states.
    threshold: float (Optional)
        Threshold beyond which to ignore the cost/similarity value.
    maximize: bool
        If false, C contain cost values to minimise. If true, C contain
        similarity values to maximise.

    Returns
    -------
    best_sorter: np.ndarray (n,)
        Sorter such that ``C[:, best_sorter]`` has minimal (resp. maximal) trace.
    costs: np.ndarray (n,)
        Array containing the cost (resp. similarity) values of the alignment.
    cost_sorter: np.ndarray (n,)
        Sorter ordering the cost (resp. similarity) values of the alignment
        from best to worse.
    """
    sign = -1 if maximize else 1
    M = C.copy()
    if threshold is not None:
        M[sign * (M - threshold) >= 0] = 0 if maximize else M.sum()
    try:
        best_sorter = linear_sum_assignment(M, maximize=maximize)[1]
    except TypeError:  # for lower scipy versions
        best_sorter = linear_sum_assignment(sign * M)[1]

    costs = np.diag(C[:, best_sorter])
    signed_costs = sign * costs
    cost_sorter = signed_costs.argsort()
    return best_sorter, costs, cost_sorter

def alignment_wasserstein_mindist_euclidean(num_markov_states: int, num_models: int, avg_mindist_locally_sorted: tuple, maximize: bool = False) -> Tuple[dict, bool]: #TODO: refactor
    """
    Align the Markov states by computing the Wasserstein distance
    over ensembles of inter-residue heavy atom minimum distance matrices.

    Parameters
    Parameters
    ----------
    num_markov_states
    num_models
    avg_mindist_locally_sorted

    Returns
    -------
    """

    uniform = np.full(num_models, 1 / num_models)
    C = np.zeros((num_markov_states, num_markov_states))
    for i in range(num_markov_states):
        for j in range(num_markov_states):
            set1 = avg_mindist_locally_sorted[0][:, i]
            set2 = avg_mindist_locally_sorted[1][:, j]
            ot_cost_mat = np.linalg.norm(set1[:, np.newaxis] - set2[np.newaxis, :], axis=-1)
            ot_plan = ot.emd(uniform, uniform, ot_cost_mat)
            C[i, j] = (ot_cost_mat * ot_plan).sum()
    return C, maximize

def perform_local_sort(system: str, num_models: int, num_markov_states: int, avg_cluster_mindists: np.ndarray) -> np.ndarray:
    """

    Execute the local alignment algorithm and update the yaml file containing the alignments.

    The yaml file corresponds to a dictionary with keys corresponding to the numbers of Markov states,
    and values are the corresponding local sorter in array format with dimension (num_models, num_markov_states).

    Parameters
    ----------
    system, str - name of the system (dataset)
    num_markov_states, int - number of Markov states for the underlying models
    num_models - number of models in the model ensemble for a given dataset
    avg_cluster_mindists, np.ndarray - shape = (num_models, num_markov_states, num_features)
                        precomputed avg features (interresidue distances) for the clusters as estimated by the models

    Returns
    -------

    """
    sorter = local_sort_mindist_kmeans(num_models=num_models, num_markov_states=num_markov_states,avg_cluster_mindists=avg_cluster_mindists)
    data = {num_markov_states: sorter.tolist()}
    filename = LOCAL_SORTERS_PATH_TEMPLATE.format(d=system)
    update_yaml(filename=filename, new_data=data)
    return sorter

def produce_alignments(args: argparse.Namespace, avg_cluster_mindists: dict, num_markov_states: int) -> None:
    """
    Function invoking both the local and the inter-system ('global') alignments. Alignments are saved into yaml files at
    designated location (as specified in config.paths.py).

    Parameters
    ----------
    args, argparse.Namespace - arguments passed by the user, mainly should contain the names of the systems at hand
    avg_cluster_mindists, dict - key = system name, value = np.ndarray with precomputed avg cluster  features (interresidue matrices)
    num_markov_states, int - Number of the Markov states for the underlying models

    Returns
    -------
    """
    assert len(args.systems) > 1, "At least 2 systems are required for the inter-system alignment."

    locally_sorted_avg_clust_mindists = {}
    for system in args.systems:
        # First locally align all systems
        local_sorter = perform_local_sort(system=system, num_models=NUM_MODELS_PER_DATASET, num_markov_states=NUM_MARKOV_STATES, avg_cluster_mindists=avg_cluster_mindists[system])
        # Apply the local sort to the mindists
        locally_sorted_avg_clust_mindists[system] = avg_cluster_mindists[system][local_sorter]

    system_pairs = [(args.systems[0], second_system) for second_system in args.systems[1:]]

    for s1,s2 in system_pairs:
        avg_mindist_locally_sorted = (locally_sorted_avg_clust_mindists[s1], locally_sorted_avg_clust_mindists[s2])
        C, maximize = alignment_wasserstein_mindist_euclidean(num_markov_states=NUM_MARKOV_STATES, num_models=NUM_MODELS_PER_DATASET, avg_mindist_locally_sorted=avg_mindist_locally_sorted)
        best_sorter, costs, cost_sorter = best_sorter(C=C, treshold=None, maximize=maximize)
        #save the system sorters
        data = {num_markov_states: best_sorter.tolist()}
        filename = SYSTEM_SORTERS_PATH_TEMPLATE.format(d=system, ref_data=REFERENCE_SYSTEM_FOR_ALIGNMENT)
        update_yaml(filename=filename, new_data=data)


    #Verify that the system sorters already contain the local sort information - should be in visualize_gradients

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', nargs='+', help='List (separated by spaces) the names of the systems for which you wish to preprocess the data.', required=True)
    args = parser.parse_args()

    avg_cluster_mindists = {}
    for system in args.systems:
        filepath = CLUSTER_AVG_PATH_TEMPLATE.format(d=system, ms=NUM_MARKOV_STATES)
        data = np.load(filepath)
        avg_cluster_mindists[system] = data
    produce_alignments(args, avg_cluster_mindists)
