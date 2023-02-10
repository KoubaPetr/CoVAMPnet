import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
from collections import defaultdict
from src.utils import triu_inverse
from config.paths import GRADIENTS_PER_JOBS_PATH_TEMPLATE, CLASSIFICATION_PER_JOBS_PATH_TEMPLATE, FEATURE_IMPORTANCE_PATH_TEMPLATE, FEATURE_IMPORTANCE_FULL_MATRIX_PATH_TEMPLATE, LOCAL_SORTERS_PATH_TEMPLATE, SYSTEM_SORTERS_PATH_TEMPLATE
from config.feature_importance_visualisation_settings import UR_CORNER_PARAMS, BOTTOM_EDGE_PARAMS, LEFT_EDGE_PARAMS, LL_CORNER_PARAMS, FIG_SIZE, CMAP, V_MIN, V_MAX, DPI
from config.data_model_params import NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, NUM_INTERRESIDUE_DISTANCES, NUM_RESIDUES, SKIPPED_DIAG_WIDTH

parser = argparse.ArgumentParser()
parser.add_argument("--frames_per_split", type=int, default=5, help="Number of frames on which the gradients were evaluated in every split.")
parser.add_argument('--num_splits', type=int, default=1, help='Number of splits (each containing frames_per_split frames.')
parser.add_argument('--reference_system', type=str, help='')
parser.add_argument('--systems', nargs='+',
                    help='List (separated by spaces) the names of the systems for which you wish to preprocess the data.',
                    required=True)


### Utils to read the precomputed data
def read_file(system: str, split: int, frames_per_split: int, data: str) -> np.ndarray:
    """
    Function for loading the precomputed gradients and classification scores (as specified by the argument "data")
    Parameters
    ----------
    system, str : name of the dataset
    split, int: id of the job computing the gradients/classification scores
    frames_per_split, int: number of frames evaluated within a job (split of frames)
    data, str: "grads" or "classifications" depending on what data are being loaded

    Returns, the loaded numpy aray
    -------

    """
    assert data in ['grads','classifications'], 'Value of data expected to be "grads" or "classifications"'
    file_path_template = GRADIENTS_PER_JOBS_PATH_TEMPLATE if data=='grads' else CLASSIFICATION_PER_JOBS_PATH_TEMPLATE
    file_path = file_path_template.format(d=system, nf=frames_per_split, jid=split)
    loaded_data = np.load(file_path)
    return loaded_data

def read_sorters(system: str, data: str, reference_system: str = None) -> list:
    """
    Function for reading the yaml files containing the sorters
    Parameters
    ----------
    system
    data
    args

    Returns
    -------

    """
    assert data in ['local','system'], 'Value of data expected to be "local" or "system"'
    sorter_path = LOCAL_SORTERS_PATH_TEMPLATE.format(d=system) if data=='local' else SYSTEM_SORTERS_PATH_TEMPLATE.format(d=system, ref_data=reference_system)

    with open(sorter_path, "r") as read:
        sorters = yaml.safe_load(read)

    return sorters

### Utils to extract the gradient values for the Markov State into which the underlying frame was classified

def extract_gradient_based_on_classification(averaged_grads: np.ndarray, argmax_idcs: np.ndarray, markov_state_grad_val: int, markov_state_classified: int) -> np.ndarray:
    """
    Take the gradients averaged over models for a given system and the preferred Markov state based on classification
    score and select the gradients for a desired markov state ('markov_state_grad_val') computed over frames classified
    into a particular Markov state ('markov_state_classified').

    Parameters
    ----------
    averaged_grads, np.ndarray - shape (NUM_MARKOV_STATES,NUM_FRAMES_TOTAL,NUM_INTERRESIDUE_DISTANCES)
    argmax_idcs, np.ndarray - shape (NUM_FRAMES_TOTAL,)
    markov_state_grad_val, int - Markov state with respect to which the gradient is computed
    markov_state_classified, int - Markov state into which the underlying frames should be classified, if None
                                   all the frames are considered

    Returns
    -------

    """
    if markov_state_classified is not None:
        frame_selection = np.where(argmax_idcs==markov_state_classified)
    else:
        frame_selection = np.where(argmax_idcs==argmax_idcs)
    extracted_grads =  averaged_grads[markov_state_grad_val,frame_selection,:].mean(axis=1)
    return extracted_grads

def main(systems: list, num_splits: int, frames_per_split: int):
    num_frames_total = num_splits*frames_per_split

    ### Read the precomputed data
    grads = defaultdict(list)
    classifications = defaultdict(list)
    local_sorters = {}
    global_sorters = {}

    for system in systems:
        for split in range(num_splits):
            grad_part = read_file(system, split, frames_per_split, 'grads')
            clas_part = read_file(system, split, frames_per_split, 'classifications')
            grads[system].append(grad_part)
            classifications[system].append(clas_part)
        grads[system] = np.concatenate(grads[system], axis=2)
        classifications[system] = np.concatenate(classifications[system], axis=2)

        ### Apply the sorters

        local_sorters[system] = read_sorters(system=system, data='local')
        if system != args.reference_system:
            global_sorters[system] = read_sorters(system=system, data='system', args=args.reference_system)
        else:
            global_sorters[system] = {NUM_MARKOV_STATES: list(range(NUM_MARKOV_STATES))}

    # Compose the locals sorter (aligning the models across a single system) and the global sorters (aligning the Markov states across systems)
    composed_sorters = {system: np.empty((NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES)) for system in systems}

    for system in systems:
        for i in range(NUM_MODELS_PER_DATASET):
            composed_sorters[system][i] = np.array(local_sorters[system][NUM_MARKOV_STATES][i], dtype='int')[global_sorters[system][NUM_MARKOV_STATES]]

    grads_sorted = {
        system: np.empty((NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, num_frames_total, NUM_INTERRESIDUE_DISTANCES)) for
        system in systems}
    class_sorted = {system: np.empty((NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, num_frames_total)) for system in
                    systems}
    for system in systems:
        for i in range(NUM_MODELS_PER_DATASET):
            grads_sorted[system][i] = grads[system][i][composed_sorters[system][i].astype('int')].copy()
            class_sorted[system][i] = classifications[system][i][composed_sorters[system][i].astype('int')].copy()

    ### Average over the NUM_MODELS_PER_DATASET models estimated for each system
    model_averaged_grads = {system: value.mean(0) for system, value in grads_sorted.items()}
    model_averaged_class_scores = {system: class_scores.mean(0) for system, class_scores in class_sorted.items()}

    ### Extract the gradient values for the Markov State into which the underlying frame was classified

    gs_per_system = {}
    gs_per_system_pooled = {}
    for system in systems:
        gs_per_system[system] = [[None for _ in range(NUM_MARKOV_STATES)] for _ in range(NUM_MARKOV_STATES)]
        gs_per_system_pooled[system] = [None for _ in range(NUM_MARKOV_STATES)]
        classification_scores = model_averaged_class_scores[system]
        argmax_idcs = np.argmax(classification_scores, axis=0)
        for ms_grad in range(NUM_MARKOV_STATES):
            for ms_class in range(NUM_MARKOV_STATES):
                gs_per_system[system][ms_grad][ms_class] = extract_gradient_based_on_classification(
                    averaged_grads=model_averaged_grads[system], argmax_idcs=argmax_idcs, markov_state_grad_val=ms_grad,
                    markov_state_classified=ms_class)
            gs_per_system_pooled[system][ms_grad] = extract_gradient_based_on_classification(
                averaged_grads=model_averaged_grads[system], argmax_idcs=argmax_idcs, markov_state_grad_val=ms_grad,
                markov_state_classified=None)

    ### Plot full matrix of gradients (rows correspond to the gradients studied w.r.t. a particular Markov state and columns correspond to the Markov states into which the underlying frames were classified)

    for system in systems:
        g = gs_per_system[system]
        fig, ax = plt.subplots(NUM_MARKOV_STATES, NUM_MARKOV_STATES, figsize=FIG_SIZE, sharex=True, sharey=True)
        fig.suptitle(system)
        for c_row in range(NUM_MARKOV_STATES):
            for c_col in range(NUM_MARKOV_STATES):
                # Select the value for normalization of the gradient values
                sup_norm = np.abs(g[c_row][c_col]).max()

                # Transform the features back into the matrix shape, considering the skipped (sub-)diagonal elements
                pcm = ax[c_row, c_col].matshow(
                    triu_inverse(g[c_row][c_col][0, :] / sup_norm, NUM_RESIDUES, n_diagonal=SKIPPED_DIAG_WIDTH),
                    cmap=CMAP, vmin=V_MIN, vmax=V_MAX)

                # Select the subfigure layout based on position in the grid
                if (c_col > 0) and (c_row < NUM_MARKOV_STATES - 1):
                    ax[c_row, c_col].tick_params(**UR_CORNER_PARAMS)
                elif (c_col > 0) and (c_row == NUM_MARKOV_STATES - 1):
                    ax[c_row, c_col].tick_params(**BOTTOM_EDGE_PARAMS)
                elif (c_col == 0) and (c_row < NUM_MARKOV_STATES - 1):
                    ax[c_row, c_col].tick_params(**LEFT_EDGE_PARAMS)
                elif (c_col == 0) and (c_row == NUM_MARKOV_STATES - 1):
                    ax[c_row, c_col].tick_params(**LL_CORNER_PARAMS)
        fig.colorbar(pcm, ax=ax)
        plt.savefig(FEATURE_IMPORTANCE_FULL_MATRIX_PATH_TEMPLATE.format(system), dpi=DPI)

    ### Plot gradients accross evaluated across all frames, disregarding their classification into Markov states

    for system in systems:
        g = gs_per_system_pooled[system]
        fig, ax = plt.subplots(NUM_MARKOV_STATES, 1, figsize=FIG_SIZE, sharex=True, sharey=True)
        fig.suptitle(system)
        for c_row in range(NUM_MARKOV_STATES):
            # Select the value for normalization of the gradient values
            sup_norm = np.abs(g[c_row]).max()

            # Transform the features back into the matrix shape, considering the skipped (sub-)diagonal elements
            triu = triu_inverse(g[c_row][0, :] / sup_norm, NUM_RESIDUES, n_diagonal=SKIPPED_DIAG_WIDTH)
            pcm = ax[c_row].matshow(triu, cmap=CMAP, vmin=V_MIN, vmax=V_MAX)

            # Select the subfigure layout based on position in the grid
            if (c_row < NUM_MARKOV_STATES - 1):
                ax[c_row].tick_params(**LEFT_EDGE_PARAMS)
            else:
                ax[c_row].tick_params(**LL_CORNER_PARAMS)
        fig.colorbar(pcm, ax=ax)
        plt.savefig(FEATURE_IMPORTANCE_PATH_TEMPLATE.format(system), dpi=DPI)

if __name__ == '__main__':
    args = parser.parse_args()
    main(systems=args.systems, num_splits=args.num_splits, frames_per_split=args.frames_per_split)