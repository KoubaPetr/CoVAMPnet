import matplotlib.pyplot as plt
import numpy as np
import yaml
from collections import defaultdict
from src.utils import triu_inverse
from config.paths import GRADIENTS_PER_JOBS_PATH_TEMPLATE, CLASSIFICATION_PER_JOBS_PATH_TEMPLATE, LOCAL_SORTERS_PATH, SYSTEM_SORTERS_PATH, FEATURE_IMPORTANCE_PATH_TEMPLATE, FEATURE_IMPORTANCE_FULL_MATRIX_PATH_TEMPLATE

### TODO change to arguments
SYSTEMS = ['ZS-ab2','ZS-ab3','ZS-ab4']
FRAMES_PER_SPLIT = 5
SPLITS = 1
NUM_MODELS_PER_DATASET = 2
NUM_MARKOV_STATES = 3
NUM_INTERRESIDUE_DISTANCES = 780
NUM_FRAMES_TOTAL = FRAMES_PER_SPLIT*SPLITS

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
    file_path = file_path_template.format(system, frames_per_split, split)
    loaded_data = np.load(file_path)
    return loaded_data

def read_sorters(data: str) -> list:
    """
    Function for reading the yaml files containing the sorters
    Parameters
    ----------
    data

    Returns
    -------

    """
    assert data in ['local','system'], 'Value of data expected to be "local" or "system"'
    sorter_path = LOCAL_SORTERS_PATH if data=='local' else SYSTEM_SORTERS_PATH

    with open(sorter_path, "r") as read:
        sorters = yaml.safe_load(read)

    return sorters

### Read the precomputed data

grads = defaultdict(list)
classifications = defaultdict(list)

for system in SYSTEMS:
    for split in range(SPLITS):
        grad_part = read_file(system,split,FRAMES_PER_SPLIT, 'grads')
        clas_part = read_file(system,split,FRAMES_PER_SPLIT, 'classifications')
        grads[system].append(grad_part)
        classifications[system].append(clas_part)
    grads[system] = np.concatenate(grads[system],axis=2)
    classifications[system] = np.concatenate(classifications[system],axis=2)

### Apply the sorters

local_sorters = read_sorters(data='local')
global_sorters = read_sorters(data='system')

#Compose the locals sorter (aligning the models across a single system) and the global sorters (aligning the Markov states across systems)
composed_sorters = {system: np.empty((NUM_MODELS_PER_DATASET,NUM_MARKOV_STATES)) for system in SYSTEMS}

for system in SYSTEMS:
    for i in range(NUM_MODELS_PER_DATASET):
        composed_sorters[system][i] = np.array(local_sorters[system][i], dtype='int')[global_sorters[system]]

grads_sorted = {system: np.empty((NUM_MODELS_PER_DATASET,NUM_MARKOV_STATES,NUM_FRAMES_TOTAL,NUM_INTERRESIDUE_DISTANCES)) for system in SYSTEMS}
class_sorted = {system: np.empty((NUM_MODELS_PER_DATASET,NUM_MARKOV_STATES,NUM_FRAMES_TOTAL)) for system in SYSTEMS}
for system in SYSTEMS:
    for i in range(NUM_MODELS_PER_DATASET):
        grads_sorted[system][i] = grads[system][i][composed_sorters[system][i].astype('int')].copy()
        class_sorted[system][i] = classifications[system][i][composed_sorters[system][i].astype('int')].copy()

### Average over the NUM_MODELS_PER_DATASET models estimated for each system
model_averaged_grads = {system: value.mean(0) for system, value in grads_sorted.items()}
model_averaged_class_scores = {system: class_scores.mean(0) for system, class_scores in class_sorted.items()}

### Extract the gradient values for the Markov State into which the underlying frame was classified

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

gs_per_system = {}
gs_per_system_pooled = {}
for system in SYSTEMS:
    gs_per_system[system] = [[None for _ in range(NUM_MARKOV_STATES)] for _ in range(NUM_MARKOV_STATES)]
    gs_per_system_pooled[system] = [None for _ in range(NUM_MARKOV_STATES)]
    classification_scores = model_averaged_class_scores[system]
    argmax_idcs = np.argmax(classification_scores,axis=0)
    for ms_grad in range(NUM_MARKOV_STATES):
        for ms_class in range(NUM_MARKOV_STATES):
            gs_per_system[system][ms_grad][ms_class] = extract_gradient_based_on_classification(averaged_grads=model_averaged_grads[system], argmax_idcs=argmax_idcs, markov_state_grad_val=ms_grad, markov_state_classified=ms_class)
        gs_per_system_pooled[system][ms_grad] = extract_gradient_based_on_classification(averaged_grads=model_averaged_grads[system], argmax_idcs=argmax_idcs, markov_state_grad_val=ms_grad, markov_state_classified=None)
    # g_00 = model_averaged_grads[system][0,np.where(idcs==0),:].mean(axis=1) #TODO package in a function
    # g_01 = model_averaged_grads[system][0,np.where(idcs==1),:].mean(axis=1)
    # g_02 = model_averaged_grads[system][0,np.where(idcs==2),:].mean(axis=1)
    # g_10 = model_averaged_grads[system][1,np.where(idcs==0),:].mean(axis=1)
    # g_11 = model_averaged_grads[system][1,np.where(idcs==1),:].mean(axis=1)
    # g_12 = model_averaged_grads[system][1,np.where(idcs==2),:].mean(axis=1)
    # g_20 = model_averaged_grads[system][2,np.where(idcs==0),:].mean(axis=1)
    # g_21 = model_averaged_grads[system][2,np.where(idcs==1),:].mean(axis=1)
    # g_22 = model_averaged_grads[system][2,np.where(idcs==2),:].mean(axis=1)
    # g_pooled_0 = model_averaged_grads[system][0,:,:].reshape(1,NUM_FRAMES_TOTAL,-1).mean(axis=1)
    # g_pooled_1 = model_averaged_grads[system][1,:,:].reshape(1,NUM_FRAMES_TOTAL,-1).mean(axis=1)
    # g_pooled_2 = model_averaged_grads[system][2,:,:].reshape(1,NUM_FRAMES_TOTAL,-1).mean(axis=1)
    # gs = [[g_00,g_01, g_02],[g_10,g_11, g_12],[g_20,g_21, g_22]]
    # gs_per_system[system] = gs
    # gs_per_system_pooled[system] = [g_pooled_0,g_pooled_1,g_pooled_2]


### FULL MATRIX OF PLOTS

for system in SYSTEMS:
    g = gs_per_system[system]
    fig, ax = plt.subplots(3, 3, figsize=(12,12), sharex=True, sharey=True)
    fig.suptitle(system)
    print('System = {}'.format(system))
    for c_row in range(NUM_MARKOV_STATES):
        for c_col in range(NUM_MARKOV_STATES):
            # scores = model_averaged_class_scores[system]
            # TODO: prepare the grid for the plots
            # TODO: apply the alignment once settled (mind the randomness coming from k-means init
            sup_norm = np.abs(g[c_row][c_col]).max()
            pcm = ax[c_row,c_col].matshow(triu_inverse(g[c_row][c_col][0,:] / sup_norm, 42, n_diagonal=2), cmap="RdBu", vmin=-1,vmax=1)
#             print('Row {}, Col {}, max value = {:.5f}'.format(c_row,c_col,(g[c_row][c_col]).max()))
#             print('Row {}, Col {}, min value = {:.5f}'.format(c_row,c_col,(g[c_row][c_col]).min()))
            if (c_col > 0) and (c_row < 2):
                ax[c_row,c_col].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    bottom='off',
                    top='off',
                    labelbottom='off',  # labels along the bottom edge are off)
                    labelleft='off'
                )
            elif (c_col > 0) and (c_row == 2):
                ax[c_row,c_col].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    bottom='on',
                    top='off',
                    labelbottom='on',  # labels along the bottom edge are off)
                    labelleft='off',
                    labeltop='off'
                )
            elif (c_col == 0) and (c_row < 2):
                ax[c_row,c_col].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='on',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    bottom='off',
                    top='off',
                    labelbottom='off',  # labels along the bottom edge are off)
                    labelleft='on'
                )
            elif (c_col == 0) and (c_row == 2):
                ax[c_row,c_col].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='on',      # ticks along the bottom edge are off
                    bottom='on',
                    right='off',         # ticks along the top edge are off
                    top='off',
                    labelbottom='on',  # labels along the bottom edge are off)
                    labelleft='on',
                    labeltop='off'
                )
    fig.colorbar(pcm,ax=ax)
    plt.savefig(FEATURE_IMPORTANCE_FULL_MATRIX_PATH_TEMPLATE.format(system), dpi=600)


### PLOTS POOLED OVER ALL FRAMES

for system in SYSTEMS:
    g = gs_per_system_pooled[system]
    fig, ax = plt.subplots(3, 1, figsize=(12,12), sharex=True, sharey=True)
    fig.suptitle(system)
    print('System = {}'.format(system))
    for c_row in range(NUM_MARKOV_STATES):
        print(g[c_row].shape)
        sup_norm = np.abs(g[c_row]).max()
        triu = triu_inverse(g[c_row][0,:] / sup_norm, 42, n_diagonal=2)
        pcm = ax[c_row].matshow(triu, cmap="RdBu", vmin=-1,vmax=1)
        if (c_row < 2):
            ax[c_row].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    bottom='off',
                    top='off',
                    labelbottom='off',  # labels along the bottom edge are off)
                    labelleft='on')
        else:
            ax[c_row].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    bottom='on',
                    top='off',
                    labelbottom='on',  # labels along the bottom edge are off)
                    labelleft='on',
                    labeltop='off')
    fig.colorbar(pcm,ax=ax)
    plt.savefig(FEATURE_IMPORTANCE_PATH_TEMPLATE.format(system), dpi=600)


if __name__ == '__main__':
    ... #TODO