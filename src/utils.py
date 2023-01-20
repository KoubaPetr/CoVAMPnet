"""
This module provides utility functions related to training a VAMPnet model on
molecular dynamics simulation data as well as analyzing its results.
"""

import numpy as np
import h5py
import os
import pyemma as pe
pe.config.mute = True
import yaml
import itertools
import mdtraj as md
from os.path import join
from scipy.linalg import eig
from glob import glob
from typing import Any, List, Optional, Sequence, Union
from src.data_fcts import DataGenerator
from scipy.stats import gaussian_kde

# system_names = ["Ab_adaptive_A14sb", "Ab_adaptive_C36m",
#                 "Ab-tmp_adaptive_C36m", "Ab-tau_adaptive_C36m",
#                 "Ab_long_C36m", "Ab-tmp_long_C36m",
#                 "Ab_adaptive_C36m_50ns", "Ab_adaptive_C36m_200ns", "ZS-ab3","ZS-ab4", "SM-ab9"]
system_names = ["ZS-ab2","ZS-ab3","ZS-ab4","SM-ab7","SM-ab8","SM-ab9",'SM-ab7-rnd','SM-ab7-sel',"ZS-ab2-sel","ZS-ab3-sel","ZS-ab4-sel"]


def unflatten(source: np.ndarray, lengths: List[np.ndarray]) -> List[np.ndarray]:
    """
    Takes an array and returns a list of arrays.

    Parameters
    ----------
    source
        Array to be unflattened.
    lengths
        List of integers giving the length of each subarray.
        Must sum to the length of source.

    Returns
    -------
    unflat
        List of arrays.
    """
    conv = []
    lp = 0
    for arr in lengths:
        arrconv = []
        for le in arr:
            arrconv.append(source[lp:le + lp])
            lp += le
        conv.append(arrconv)
    ccs = list(itertools.chain(*conv))
    return ccs


def triu_inverse(x: np.ndarray, N: int, n_diagonal: int = 0) -> np.ndarray:
    """
    Converts flattened upper-triangular matrices with zero off-diagonal terms into full
    symmetric matrices.

    Parameters
    ----------
    x: np.ndarray
        Flattened matrices of size (length, -1) or matrix of size (-1,)
    N: int
        Size of the N * N matrix
    n_diagonal: int
        Number of off-diagonal terms that are set to 0

    Returns
    -------
    mat: np.ndarray
        Array of shape (length, N, N) or (N, N)
    """
    if x.ndim == 1:
        mat = np.zeros((N, N))
    else:
        mat = np.zeros((x.shape[0], N, N))
    list_indices = list(zip(*np.triu_indices(N, k=1)))
    list_indices = [i for i in list_indices if np.abs(i[0] - i[1]) > n_diagonal]
    a = np.array([i[0] for i in list_indices])
    b = np.array([i[1] for i in list_indices])
    mat[..., a, b] = x
    mat += mat.swapaxes(-2, -1)
    return mat


def statdist(X: np.ndarray) -> np.ndarray:
    """
    Calculate the equilibrium distribution of a transition matrix.

    Parameters
    ----------
    X
        Row-stochastic transition matrix

    Returns
    -------
    mu
        Stationary distribution, i.e. the left
        eigenvector associated with eigenvalue 1.
    """
    ev, evec = eig(X, left=True, right=False)
    mu = evec.T[ev.argmax()]
    mu /= mu.sum()
    return mu


def compute_lag(lag_times: Union[float, Sequence], dt: float) -> Union[int, np.ndarray]:
    """
    Convert lag times to lag steps.

    Parameters
    ----------
    lag_times
        Number or sequence of lag time(s).
    dt
        Timestep of the trajectories with the same unit as lag_times.

    Returns
    -------
    lags
        Number of steps or sequence of number of steps corresponding to the
        input lag times.
    """
    if isinstance(lag_times, (np.ndarray, list, tuple)):
        lag_times = np.array(lag_times)
        return (lag_times / dt).astype(int)
    return int(lag_times / dt)


def get_outsizes(filename: str, k: str) -> np.ndarray:
    """
    Find the numbers of clusters used by the model from the results file.

    Parameters
    ----------
    filename: str
        Location of the results file.
    k: str
        Name of the system.

    Returns
    -------
    outsizes: array
        Array of number of clusters.
    """
    with h5py.File(filename, "r") as read:
        outsizes = [int(n) for n in read[k]['0'].keys()]
    return np.array(outsizes, dtype=int)


def get_attempts(filename: str, k: str) -> int:
    """
    Find the number of attempts used by the model from the results file.

    Parameters
    ----------
    filename: str
        Location of the results file.
    k: str
        Name of the system.

    Returns
    -------
    attempts: int
        Number of attempts.
    """
    with h5py.File(filename, "r") as read:
        attempts = len(read[k].keys())
    return attempts


def get_new_outsizes(run_path: str, k: str) -> np.ndarray:
    """
    Find the numbers of clusters remaining to be estimated by a model.

    Parameters
    ----------
    run_path: str
        Location of the run directory.
    k: str
        Name of the system.

    Returns
    -------
    outsizes: array
        Array of numbers of clusters.
    """
    filename = os.path.join(run_path, "results/data.hdf5")
    outsizes = np.loadtxt(os.path.join(run_path, "outsizes.txt"), dtype=int)
    try:
        old_outsizes = get_outsizes(filename, k)
        new_outsizes = np.setdiff1d(outsizes, old_outsizes)
        return new_outsizes
    except OSError:
        return np.array(outsizes, dtype=int).reshape(-1)  # ensure 1D array


def get_last_step_model_path(run_path: str, k: str, n: int,
                             attempt: int) -> str:
    """
    Find the location of the last checkpoint of a model for a given number
    of clusters and attempt.

    Parameters
    ----------
    run_path: str
        Location of the run directory.
    k: str
        Name of the system.
    n: int
        Number of clusters.
    attempt: int
        Number of attempts.

    Returns
    -------
    filename: str
        Location of the model checkpoint.
    """
    filename = join(run_path, "models/model-ve-{0}-{1}-{2}-intermediate-{{}}.hdf5".format(k, n, attempt))
    step = 0
    if not os.path.isfile(filename.format(step)):
        raise FileNotFoundError
    while True:
        if not os.path.isfile(filename.format(step + 1)):
            return filename.format(step)
        step += 1


def update_nested_dict(old_d: dict, new_d: dict) -> dict:
    """
    Recursively update a nested dictionary.

    Parameters
    ----------
    old_d: dict
        Current dictionary.
    new_d: dict
        Dictionary to update old_d with.

    Returns
    -------
    d: dict
        Updated dictionary.
    """
    d = old_d.copy()
    for key, value in new_d.items():
        if key in d and type(d[key]) is dict and type(value) is dict:
            d[key] = update_nested_dict(d[key], value)
        else:
            d[key] = value
    return d


def load_yaml(filename: str) -> dict:
    """
    Load a yaml file.

    Parameters
    ----------
    filename: str
        Location of the file.

    Return
    -------
    data: dict
        Loaded data.
    """
    with open(filename, "r") as read:
        data = yaml.safe_load(read)
    return data


def update_yaml(filename: str, new_data: dict) -> None:
    """
    Update a yaml file with new data. If no file exists then the data is simply
    saved.

    Parameters
    ----------
    filename: str
        Location of the file.
    new_data: dict
        New data to save.
    """
    try:
        with open(filename, "r") as read:
            data = yaml.safe_load(read)
        assert type(data) is dict
        data = update_nested_dict(data, new_data)
    except (AssertionError, FileNotFoundError):
        data = new_data

    with open(filename, "w") as write:
        yaml.safe_dump(data, write, default_flow_style=None)


class Utils:
    """Base class for printing information depending on verbosity."""

    def _print(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)


class Quantities(Utils):
    nres = 42
    # base_simulations_folder = os.path.join(os.getcwd(),"simulations")

    def __init__(
            self,
            base_run_path: str,
            k: str,
            k_other: str,
            attempts: Optional[int] = None,
            outsizes: Optional[Union[List, np.ndarray]] = None,
            md_data: str = "sergio",
            train: bool = False,
            verbose: bool = True
    ) -> None:
        """
        Wrapper to dynamically load and compute all quantities of interest
        in the training of VAMPnets and the analysis of their results.

        Parameters
        ----------
        base_run_path: str
            Location of the run base directory.
        k: str
            Name of the system.
        k_other: str
            Name of the other system w.r.t. which we are aligning, used for pooled TICA.
        attempts: int (optional)
            Number of training attempts. If not specified it is loaded from
            the training results file.
        outsizes: list or array (optional)
            Numbers of Markov states. If not specified they are loaded from
            the training results file.
        md_data: str
            Source of the molecular dynamics simulation data to use.
        train: bool
            If true, do not try to load attempts or outsizes from file.
        verbose: bool
            Verbosity.
        """
        self.base_run_path = base_run_path
        self.k = k
        self.k_other = k_other # used for the pooled TICA
        self.verbose = verbose
        self.md_data = md_data

        # Check if data origin correctly specified
        if self.md_data not in ["sergio", "zainab", "loehr"]:
            raise ValueError("MD simulation data specified ({0}) is unknown.".format(self.md_data))

        # List MD trajectory and topology files
        # self.simulation_folder = join(self.base_simulations_folder, self.md_data)
        if self.md_data == "sergio":
            self.run_path = join(self.base_run_path, self.k)
            self.trajs = sorted(glob(join(self.run_path, 'simulations', "310k_md*/filtered.ALL.xtc")))
            self.trajs_dict = {path.split('/')[-2]: path for path in glob(join(self.run_path, 'simulations', "310k_md*/filtered.ALL.xtc"))}
            self.top = join(self.run_path, 'simulations', "filtered.pdb")
            # if "long" in self.k:
            #     self.trajs = sorted(glob(join(self.simulation_folder, self.k, "filtered/310k_md*/filtered.ALL.xtc")))
            # else:
            #     self.trajs = sorted(glob(join(self.simulation_folder, self.k, "filtered/ab*/output.filtered.xtc")))
            self.top = join(self.run_path, 'simulations', "filtered.pdb")
        elif self.md_data == "zainab":
            self.run_path = join(self.base_run_path, self.k)
            self.trajs = sorted(glob(join(self.run_path, 'simulations', "e*s*_*/output.filtered.xtc")))
            self.trajs_dict = {path.split('/')[-2]: path for path in glob(join(self.run_path, 'simulations', "e*s*_*/output.filtered.xtc"))}
            self.top = join(self.run_path, 'simulations', "filtered.pdb")
        # elif self.md_data == "loehr":
        #     self.run_path = self.base_run_path  # only trained models with reduced abeta
        #     self.trajs = sorted(glob(join(self.simulation_folder, self.k, "r?/traj*.xtc")))
        #     self.top = join(self.simulation_folder, self.k, "topol.gro")
        self.topo = md.load_topology(self.top)

        # Important directories
        self.data_source = join(self.run_path, "results/data.hdf5")
        self.training_path = join(self.run_path, "training")
        self.models_path = join(self.run_path, "models")
        self.generators_path = self.models_path

        # Load attempts and outsizes
        self._print("System:  ", self.k)
        if not train:
            if attempts is not None:
                self.attempts = attempts
            else:
                self.attempts = get_attempts(self.data_source, k)
            self._print("Attempts:", self.attempts)

            if outsizes is not None:
                self.outsizes = outsizes
            else:
                self.outsizes = get_outsizes(self.data_source, k)
            self._print("Outsizes:", self.outsizes)

    @property
    def sorters(self) -> None:
        """Local alignment sorters."""
        try:
            return self._sorters
        except AttributeError:
            raise AttributeError("No sorters defined.")

    @sorters.setter
    def sorters(self, sorters: dict) -> None:
        """
        Set or update the local alignment sorters.

        If previous sorters were already set, then delete all other quantities
        that depend on it.

        Parameters
        ----------
        sorters: dict
            Local alignment sorters.
        """
        self._sorters = sorters
        for attr in ["pfs", "koops", "pis", "conws_sorted", "contacts", "sec"]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def __getattr__(self, attr: str) -> Any:
        # Configuration file
        if attr in ["cfg", "ratio", "nnargs", "max_frames"]:
            self.cfg = load_yaml(join(self.base_run_path, "config.yaml"))
            self.ratio = self.cfg["training"]["ratio"]
            self.nnargs = self.cfg["training"]["nnargs"]
            self.max_frames = self.cfg["training"]["max_frames"][self.k]
        elif attr in ["lag", "analysis_lag", "network_lag", "steps", "lags"]:
            self.lag = compute_lag(self.cfg["analysis"]["lag_ns"], self.dt)
            self.analysis_lag = self.lag
            self.network_lag = compute_lag(self.cfg["training"]["lag_ns"], self.dt)
            self.steps = self.cfg["analysis"]["ck_steps"]
            self.lags = compute_lag(self.cfg["analysis"]["it_lags_ns"], self.dt)


        # Raw trajectories
        elif attr in ["dt", "nframes", "lengths"]:
            filename = join(self.base_run_path, self.k, 'simulations', "info.yml")
            # Load quantity if possible, else compute it then save it
            # dt has to be put manually in the file, as the one from MDtraj
            # has inconsistent magnitude
            try:
                setattr(self, attr, load_yaml(filename)[attr])
            except (KeyError, FileNotFoundError) as e:
                if attr == "nframes":
                    self.nframes = self.inpcon.trajectory_lengths().sum()
                    data = {"nframes": int(self.nframes)}
                elif attr == "lengths":
                    self.lengths = [self.inpcon.trajectory_lengths()]
                    data = {"lengths": [array.tolist() for array in self.lengths]}
                else:
                    raise e
                update_yaml(filename, data)
        elif attr in ["inpcon", "lengths"]:
            self._print("Loading trajectories")
            feat = pe.coordinates.featurizer(self.top)
            feat.add_residue_mindist()
            self.inpcon = pe.coordinates.source(self.trajs, feat)

        # Preprocessed data
        elif attr == "raw":
            filename = join(self.base_run_path,self.k,'simulations', "intermediate",
                            "mindist-780-{0}.npy".format(self.k))
            if os.path.isfile(filename):
                self._print("Loading {0}".format(filename))
                self.raw = np.load(filename)
            else:
                self._print("Computing features and saving at {0}".format(filename))
                self.raw = np.vstack(self.inpcon.get_output())
                np.save(filename, self.raw)
        elif attr in ["input_data", "n_dims"]:
            raw_mean, raw_std = self.raw.mean(axis=0), self.raw.std(axis=0)
            self.input_data = [(r - raw_mean) / raw_std
                               for r in unflatten(self.raw, self.lengths)]
            self.n_dims = self.raw.shape[1]
            del self.raw
        elif attr in ["mindist_flat", "mindist"]:
            filename = join(self.base_run_path,self.k,'simulations', "intermediate",
                            "mindist-all-{0}.npy".format(self.k))
            if os.path.isfile(filename):
                self._print("Loading {0}".format(filename))
                self.mindist_flat = np.load(filename)
            else:
                self._print("Computing features"
                            "and saving at {0}".format(filename))
                allpairs = np.asarray(list(itertools.combinations(range(self.nres), 2)))
                feat = pe.coordinates.featurizer(self.top)
                feat.add_residue_mindist(residue_pairs=allpairs)
                inpmindist = pe.coordinates.source(self.trajs, feat)
                self.mindist_flat = np.vstack(inpmindist.get_output())
                np.save(filename, self.mindist_flat)
            self.mindist = unflatten(self.mindist_flat, self.lengths)

            #save mindist (full not just 780)
            filename = join(self.base_run_path,self.k,'simulations', "intermediate",
                            "mindist-{0}.npy".format(self.k))
            np.save(filename, self.mindist_flat)


        # VAMPnet output
        elif attr == "pf":
            self._print("Loading pf")
            self.pf = {n: np.empty((self.attempts, self.nframes, n))
                       for n in self.outsizes}
            with h5py.File(self.data_source, "r") as read:
                for n in self.outsizes:
                    for i in range(self.attempts):
                        self._print("Loading k={0} i={1} n={2}".format(self.k, i, n), end="\r")
                        self.pf[n][i] = read["{0}/{1}/{2}/full".format(self.k, i, n)][:, :n]
            self._print()
        elif attr == "pfs":
            self._print("Loading pfs")
            self.pfs = {n: np.empty((self.attempts, self.nframes, n)) for n in self.outsizes}
            with h5py.File(self.data_source, "r") as read:
                for n in self.outsizes:
                    for i in range(self.attempts):
                        self._print("Loading k={0} i={1} n={2}".format(self.k, i, n), end="\r")
                        pf = read["{0}/{1}/{2}/full".format(self.k, i, n)][:, :n]
                        self.pfs[n][i] = pf[:, self.sorters[n][i]]
        elif attr == "pfsn":
            self._print("Loading pfsn")
            self.pfsn = {n: np.empty_like(self.pfs[n]) for n in self.outsizes}
            for n in self.outsizes:
                for i in range(self.attempts):
                    self.pfsn[n][i] = self.pfs[n][i] / self.pfs[n][i].sum(axis=0)
        elif attr == "koops":
            self._print("Loading koops")
            self.koops = {n: np.empty((self.attempts, n, n)) for n in self.outsizes}
            with h5py.File(self.data_source, "r") as read:
                for n in self.outsizes:
                    for i in range(self.attempts):
                        data = read["{0}/{1}/{2}/k".format(self.k, i, n)]
                        self.koops[n][i] = data[:][self.sorters[n][i]][:, self.sorters[n][i]]
        elif attr == "pis":
            self._print("Loading pis")
            self.pis = {n: np.empty((self.attempts, n)) for n in self.outsizes}
            for n in self.outsizes:
                for i in range(self.attempts):
                    self.pis[n][i] = statdist(self.koops[n][i])

        # Weights
        elif attr == "weights":
            self.weights = {}
            for n in self.outsizes:
                self.weights[n] = np.empty((self.attempts, self.nframes))
                for i in range(self.attempts):
                    w = self.pfs[n][i] @ self.pis[n][i]
                    self.weights[n][i] = w / w.sum()


        ### below is necessary if we want to have the pooled TICA in the first go, without one of the systems precomputed
        elif attr in ["nres_other","top_other","trajs_other"]:
            raise NotImplementedError

        elif attr == "lengths_other":
            self.lengths_other = yaml.safe_load(open(os.path.join(self.base_run_path,self.k_other,'simulations', "info.yml")))['lengths']


        elif attr == "mindist_other_system":
            assert self.k_other in system_names, "Name of the second system was not provided correctly, TICA over pooled data cannot be performed"

            filename_other = join(self.base_run_path,self.k_other,'simulations', "intermediate",
                            "mindist-{0}.npy".format(self.k_other))
            if os.path.isfile(filename_other):
                self._print("Loading {0}".format(filename_other))
                self.mindist_flat_other = np.load(filename_other)
            else:
                self._print("Implicit precomputing of the other system is not implemented, try analyzing the other system first to precompute the mindists for it")

                self._print("Computing features"
                            "and saving at {0}".format(filename_other))
                allpairs = np.asarray(list(itertools.combinations(range(self.nres_other), 2)))
                feat = pe.coordinates.featurizer(self.top_other)
                feat.add_residue_mindist(residue_pairs=allpairs)
                inpmindist = pe.coordinates.source(self.trajs_other, feat)
                self.mindist_flat_other = np.vstack(inpmindist.get_output())
                np.save(filename_other, self.mindist_flat_other)
            self.mindist_other_system = unflatten(self.mindist_flat_other, self.lengths_other)

        elif attr == "mindist_pooled": ### careful mindist_pooled is a tuple containing the separation index of both systems
            mindist_copy = self.mindist.copy()
            mindist_copy.extend(self.mindist_other_system.copy())
            self.mindist_pooled = mindist_copy, len(self.mindist_flat)

        # TICA
        elif attr in ["ticacon", "ycon", "ticacon_output"]:
            self._print("Computing TICA")
            self.ticacon = pe.coordinates.tica(self.mindist,
                                               lag=compute_lag(1, self.dt),
                                               dim=-1, kinetic_map=True)
            ticscon = self.ticacon.get_output()
            self.ticacon_output = ticscon
            self.ycon = np.vstack(ticscon)

        elif attr in ["F01","F23"]: #Free energy computed using KDE (kernel density estimation). Based on Crooks theorem: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.61.2361
            if attr == "F01":
                tIC1, tIC2 = 0,1
            elif attr == "F23":
                tIC1, tIC2 = 2,3

            KBT = 2.6713733112649497  # TODO make this constant more global

            subsample = 10
            kernel = gaussian_kde(self.ycon[::subsample, tIC1:tIC2 + 1].T)
            xmin, ymin, *_ = self.ycon.min(axis=0)
            xmax, ymax, *_ = self.ycon.max(axis=0)
            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            posi_stacked = np.vstack((X.ravel(), Y.ravel()))
            Z = kernel(posi_stacked).reshape(X.shape)
            mat = np.rot90(Z.copy())
            mat[mat < 0.01] = np.nan

            F = -KBT * np.log(Z)
            F -= F.min()

            if attr == "F01":
                self.F01 = F, X, Y
            elif attr == "F23":
                self.F23 = F, X, Y




        elif attr in ["ticacon_pooled", "ycon_pooled"]:
            self._print("Computing TICA pooled")
            self._print("Warning! Assuming both systems have same lag time dt !!!")

            #read pooled data
            data_pooled, separation_idx = self.mindist_pooled

            #compute TICA over pooled data
            self.ticacon_pooled = pe.coordinates.tica(data_pooled,
                                               lag=compute_lag(1, self.dt),
                                               dim=-1, kinetic_map=True)
            ticscon_pooled = self.ticacon_pooled.get_output()
            ticscon_pooled_array = np.vstack(ticscon_pooled)

            #separate based on systems #TODO check it does not get mixed up
            ycon_pooled_sys1 = ticscon_pooled_array[:separation_idx,:]
            ycon_pooled_sys2 = ticscon_pooled_array[separation_idx:,:]

            self.ycon_pooled = {self.k: ycon_pooled_sys1, self.k_other: ycon_pooled_sys2}


        # Alignment features
        elif attr == "conws":
            self._print("Computing cluster average mindist")
            num_features = self.mindist_flat.shape[1]
            self.conws = {n: np.empty((self.attempts, n, num_features))
                          for n in self.outsizes}
            for i in range(self.attempts):
                for n in self.outsizes:
                    self._print("Processing k={0} i={1} n={2}".format(self.k, i, n), end="\r")
                    self.conws[n][i] = (self.pf[n][i] / self.pf[n][i].sum(axis=0)).T @ self.mindist_flat
            self._print()
        elif attr == "conws_sorted":
            self.conws_sorted = {n: np.empty_like(self.conws[n])
                                 for n in self.outsizes}
            for i in range(self.attempts):
                for n in self.outsizes:
                    self.conws_sorted[n][i] = self.conws[n][i][self.sorters[n][i]]

        # Secondary structure
        elif attr == "dssplow":
            filename = join(self.base_run_path,self.k,'simulations',"intermediate/dssplow-{0}.npy".format(self.k))
            if os.path.isfile(filename):
                self._print("Loading dssplow")
                self.dssplow = np.load(filename)
            else:
                self._print("Computing dssplow")
                # Custom func makes this a lot easier
                dssptable = str.maketrans("HBEGITS ", "01234567") #Replaces (with numbers) codes for structures used in https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html

                def dssp_enc(traj):
                    table, _ = self.topo.to_dataframe()
                    lower_bound = table[table.resName == 'ASP'].index.min()
                    upper_bound = table[table.resName == 'ALA'].index.max() + 1
                    raw = md.compute_dssp(traj.atom_slice(range(lower_bound, upper_bound)), simplified=False)
                    return np.char.translate(raw, table=dssptable).astype(np.float32)

                feat = pe.coordinates.featurizer(self.top)
                feat.add_custom_func(dssp_enc, dim=42)
                inp = pe.coordinates.source(self.trajs, feat)
                dssp = np.vstack(inp.get_output()).astype(np.int32)

                # One-hot encoding
                nvals = dssp.max() + 1
                dsspoh = np.eye(nvals, dtype=np.int32)[dssp]

                # We could use the simplified DSSP scheme, but this gives us a bit more flexibility
                self.dssplow = np.empty((self.nframes, self.nres, 4))
                self.dssplow[:, :, 0] = dsspoh[:, :, [0, 3, 4]].sum(axis=-1)
                self.dssplow[:, :, 1] = dsspoh[:, :, [1, 2]].sum(axis=-1)
                self.dssplow[:, :, 2] = dsspoh[:, :, [5, 6]].sum(axis=-1)
                self.dssplow[:, :, 3] = dsspoh[:, :, 7]

                np.save(filename, self.dssplow)

        # Cluster-averaged secondary structure
        elif attr == "sec":
            self._print("Computing secondary structures")
            inds = unflatten(np.arange(self.nframes).reshape(-1, 1), self.lengths)
            self.sec = {n: np.empty((self.attempts, n, self.nres, 4)) for n in self.outsizes}
            for i in range(self.attempts):
                generator = DataGenerator.from_state(inds, join(self.generators_path,
                                                                "model-idx-{0}-{1}.hdf5".format(self.k, i)))
                idx = generator(n=min(self.outsizes), lag=self.network_lag).trains[0].flatten().astype(int) #Whats up with these idcs - pfs should not have such dimension
                for n in self.outsizes:
                    self._print("Processing k={0} n={1} i={2}...".format(self.k, n, i), end="\r")
                    self.sec[n][i] = (np.einsum("ij,ikl->jkl", self.pfs[n][i, idx], self.dssplow[idx]) /
                                      self.pfs[n][i, idx].sum(axis=0).reshape(-1, 1, 1))
            self._print()

        # Contact maps
        elif attr == "contacts":
            self._print("Computing contact maps")
            inds = unflatten(np.arange(self.nframes).reshape(-1, 1), self.lengths)
            cutoff = 0.8
            self.contacts = {n: np.empty((self.attempts, n, self.nres, self.nres)) for n in self.outsizes}
            for i in range(self.attempts):
                generator = DataGenerator.from_state(inds, join(self.generators_path,
                                                                "model-idx-{0}-{1}.hdf5".format(self.k, i)))
                idx = generator(n=min(self.outsizes), lag=self.network_lag).trains[0].flatten().astype(int)
                for n in self.outsizes:
                    self._print("Processing k={0} n={1} i={2}...".format(self.k, n, i), end="\r")
                    con = (np.einsum("jk,jl->kl", self.pfs[n][i, idx], (self.mindist_flat[idx] < cutoff)) /
                           self.pfs[n][i, idx].sum(axis=0).reshape(-1, 1))
                    self.contacts[n][i] = np.asarray([triu_inverse(con[j, :, ][None, :], self.nres)[0] for j in
                                                      range(n)])  # changed removed con[j] to con[j,:,][None,:]
            self._print()

        # Entropies
        elif attr == "ents":
            self.ents = {}
            for i, n in enumerate(self.outsizes):
                ent = -np.nansum(self.pfsn[n] * np.log2(self.pfsn[n]) / np.log2(self.pfsn[n].shape[1]), axis=1)
                self.ents[n] = np.array([ent.mean(axis=0), *np.percentile(ent, (2.5, 97.5), axis=0)])

        return super().__getattribute__(attr)


class QUtils:
    """Base class with Quantities dependency."""

    def __getattr__(self, attr: str) -> Any:
        if hasattr(self.quantities, attr):
            return getattr(self.quantities, attr)
        raise AttributeError("'{0}' object has no attribute '{1}' and neither does its 'Quantities' dependency".format(
            self.__class__.__name__, attr))


class QUtilsSorted(QUtils):
    def __init__(
            self,
            base_run_path: str,
            k: str,
            k_other: str = None,
            attempts: Optional[int] = None,
            outsizes: Optional[Union[List, np.ndarray]] = None,
            md_data: str = "sergio",
            system_sorter: Optional[dict] = None,
            verbose: bool = True
    ) -> None:
        """
        Utility loading automatically local and global sorters.

        Parameters
        ----------
        base_run_path: str
            Location of the run base directory.
        k: str
            Name of the system.
        k_other: str
            Name of the other system w.r.t. which we are aligning, used for pooled TICA.
        attempts: int (optional)
            Number of training attempts. If not specified it is loaded from
            the training results file.
        outsizes: list or array (optional)
            Numbers of Markov states. If not specified they are loaded from
            the training results file.
        md_data: str
            Source of the molecular dynamics simulation data to use.
        system_sorter: dict (optional)
            Global alignment sorter.
        verbose: bool
            Verbosity.
        """
        self.base_run_path = base_run_path
        self.k = k
        self.k_other = k_other
        self.verbose = verbose
        print(md_data)
        self.quantities = Quantities(self.base_run_path, self.k, k_other = self.k_other,
                                     attempts=attempts, outsizes=outsizes,
                                     md_data=md_data, verbose=self.verbose)



        if md_data=='sergio':
            md_data_token = 'sm'
        elif md_data=='zainab':
            md_data_token = 'zs'
        else:
            raise ValueError('Unexpected value md_data = {}'.format(md_data))

        sys_name_to_num = {'ZS-ab2':2, 'ZS-ab3':3, 'ZS-ab4':4, 'SM-ab7':7, 'SM-ab8':8, 'SM-ab9':9, 'SM-ab7-rnd':7, 'SM-ab7-sel':7, "ZS-ab2-sel":2,"ZS-ab3-sel":3,"ZS-ab4-sel":4}
        sys_num_0 = min(sys_name_to_num[self.k], sys_name_to_num[self.k_other])
        sys_num_1 = max(sys_name_to_num[self.k], sys_name_to_num[self.k_other])


        self.alignment_path = join(base_run_path, "alignment/align_{}_{}v{}".format(md_data_token,sys_num_0,sys_num_1))
        # Local alignment
        try:
            self.sorters
        except AttributeError:
            sorters_path = join(self.alignment_path, "local_sorters.yml")
            assert os.path.isfile(sorters_path), "Local alignment could not be found."
            self.quantities.sorters = load_yaml(sorters_path)[self.k]

        # Global alignment
        self.load_global_sorter(system_sorter=system_sorter)

    def load_global_sorter(self, system_sorter: Optional[dict] = None) -> None:
        """Load the global alignment sorter. It can be given as input, else it
        is loaded from file, and if it cannot be found from file then an identity
        sorter is used instead."""
        system_sorters_path = join(self.alignment_path, "system_sorters.yml")
        if system_sorter is not None:
            self.global_sorter = system_sorter
        elif os.path.isfile(system_sorters_path):
            self.global_sorter = load_yaml(system_sorters_path)[self.k]
        else:
            self.global_sorter = {n: np.arange(n) for n in self.outsizes}
