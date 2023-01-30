import numpy as np
import pyemma as pe
from src.utils import update_yaml
from config.paths import TRAJECTORY_PATHS_TEMPLATE, TOPOLOGY_PATH_TEMPLATE, MINDIST_PATH_TEMPLATE, INFO_PATH_TEMPLATE
from glob import glob
from argparse import ArgumentParser

def preprocess_trajectories(dataset_name: str = None, nres: int = None) -> None:
    """

    Load the .xtc trajectories, under assumption that all data is placed in the locations as described in the project README.
    It saves the precomputed interresidue distances in a single numpy array.

    Parameters
    ----------
    dataset_name: str, name of the dataset - it should be exactly how the directories immediatly holding its data are called
    nres: int, number of residues of the protein, only needs to be supplied if we would like to precompute the allresidue pair features

    Returns
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--systems', nargs='+', help='List (separated by spaces) the names of the systems for which you wish to preprocess the data.', required=True)
    parser.add_argument('--nres', type=int, default=42, help='Number of residues of the studied protein')
    args = parser.parse_args()

    for system in args.systems:
        preprocess_trajectories(system, nres=args.nres)
