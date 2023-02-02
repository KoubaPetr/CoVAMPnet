import numpy as np
import pyemma as pe
import h5py
from src.utils import update_yaml
from config.paths import TRAJECTORY_PATHS_TEMPLATE, TOPOLOGY_PATH_TEMPLATE, MINDIST_PATH_TEMPLATE, INFO_PATH_TEMPLATE, CLUSTER_AVG_PATH_TEMPLATE, MODEL_OUTPUTS_PATH_TEMPLATE
from config.data_model_params import NUM_MARKOV_STATES, NUM_MODELS_PER_DATASET
from glob import glob
from argparse import ArgumentParser

def read_classification_scores(system: str, num_models: int, num_markov_states:int, total_frames: int):
    classification_probs = np.empty((num_models, total_frames, num_markov_states))
    data_path = MODEL_OUTPUTS_PATH_TEMPLATE.format(d=system)
    h5py_key = system+'-sel' #The name of the group in the hdf5 file was modified to -sel, to denote we are dealing with the preselected model, in the multiattempt training setting
    with h5py.File(data_path, "r") as read:
        for i in range(num_models):
            classification_probs[i] = read["{0}/{1}/{2}/full".format(h5py_key, i, num_markov_states)][:, :num_markov_states][:total_frames,:] #TODO: right now this might be missmatching - check with correct data

    return classification_probs

def preprocess_trajectories(dataset_name: str = None, nres: int = None) -> tuple:
    """

    Load the .xtc trajectories, under assumption that all data is placed in the locations as described in the project README.
    It saves the precomputed interresidue distances in a single numpy array.

    Parameters
    ----------
    dataset_name: str, name of the dataset - it should be exactly how the directories immediatly holding its data are called
    nres: int, number of residues of the protein, only needs to be supplied if we would like to precompute the allresidue pair features

    Returns, tuple[np.ndarray, list[int]] - np.array corresponding to flattened mindist and a list with lengths of the trajectories for a given dataset
    -------

    """
    ### Prepare the path data for the particular dataset
    trajectory_paths = TRAJECTORY_PATHS_TEMPLATE.format(dataset_name)
    topology_path = TOPOLOGY_PATH_TEMPLATE.format(dataset_name)
    mindist_output_path = MINDIST_PATH_TEMPLATE.format(dataset_name)
    info_output_path = INFO_PATH_TEMPLATE.format(dataset_name)

    ### Read the trajectories and prepare the interresidue distance matrices
    trajs = sorted(glob(trajectory_paths))

    ''' 
    # Code which could be used in case all atom pairs are necessary
    
        import itertools
        allpairs = np.asarray(list(itertools.combinations(range(nres), 2)))
        feat = pe.coordinates.featurizer(topology_path)
        feat.add_residue_mindist(residue_pairs=allpairs)
        inpmindist = pe.coordinates.source(trajs, feat)
    '''

    feat = pe.coordinates.featurizer(topology_path)
    feat.add_residue_mindist()
    inpmindist = pe.coordinates.source(trajs, feat)

    ### Update the information about the data
    lengths = list(inpmindist.trajectory_lengths())
    update_yaml(filename=info_output_path, new_data={'lengths': [int(l) for l in lengths]})

    ### Save the preprocessed trajectories
    # This flattened data can be used together with the 'info.yml' and src.utils.unflatten()
    mindist_flat = np.vstack(inpmindist.get_output())
    np.save(file=mindist_output_path, arr=mindist_flat)

    return mindist_flat, lengths

def compute_cluster_avg_mindist(system: str, num_models: int, num_markov_states: int, mindist_flat: np.ndarray, classification_probs: np.ndarray) -> np.ndarray: #classification probs = self.pf[n]
    """
    Compute the average frame representation (interresidue distances) per Markov state (cluster)

    Parameters
    ----------
    system, str - name of the system (dataset)
    num_models, int - how many models are estimated per system
    num_markov_states, int - how many Markov states in the estimated models
    mindist_flat, np.ndarray - flattened interresidue distances for frames in the dataset
    classification_probs, Markov state probabilities for each frame, based on the models at hand

    Returns, np.ndarray - shape = (NUM_MODELS, NUM_MARKOV_STATES, NUM_INTERRESIDUE_DISTANCES)
    -------

    """
    num_features = mindist_flat.shape[1]
    avg_mindists = np.empty((num_models, num_markov_states, num_features))
    for i in range(num_models):
        avg_mindists[i] = (classification_probs[i] / classification_probs[i].sum(axis=0)).T @ mindist_flat

    outfile_path = CLUSTER_AVG_PATH_TEMPLATE.format(d=system, ms=num_markov_states)
    np.save(arr=avg_mindists, file=outfile_path)
    return avg_mindists

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--systems', nargs='+', help='List (separated by spaces) the names of the systems for which you wish to preprocess the data.', required=True)
    parser.add_argument('--nres', type=int, default=42, help='Number of residues of the studied protein')
    args = parser.parse_args()

    for system in args.systems:
        mindist_flat, traj_lengths = preprocess_trajectories(system, nres=args.nres)
        total_frames_in_dataset = sum(traj_lengths)
        classification_probs = read_classification_scores(system=system, num_models=NUM_MODELS_PER_DATASET, num_markov_states=NUM_MARKOV_STATES,total_frames=total_frames_in_dataset)
        _ = compute_cluster_avg_mindist(system, num_models=NUM_MODELS_PER_DATASET, num_markov_states=NUM_MARKOV_STATES, mindist_flat=mindist_flat, classification_probs=classification_probs)

