# Comparative Markov State Analysis (CoVAMPnet)

Code supplementing the paper "Effects of Alzheimer’s Disease Drug Candidates on Disordered Aβ42 Dissected by Comparative Markov State Analysis CoVAMPnet" [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.01.06.523007v1)

For training the VAMPnet and for performing the analyses over a single system, we acknowledge and refer to the [Ab-42 Kinetic ensemble repo](https://github.com/vendruscolo-lab/ab42-kinetic-ensemble).

For the Comparative Markov State Analysis (CoVAMPnet), we provide the code in this repository and supplementing data [on project web page (17 GB)](https://data.ciirc.cvut.cz/public/projects/2023CoVAMPnet/covampnet_data.tar.gz).

## Running CoVAMPnet

### Requirements

The user is assumed to have `conda` installed.

To download the repository, in the terminal run:
```bash
git clone git@github.com:KoubaPetr/CoVAMPnet.git 
```

To create and install the conda environment with the necessary packages, run:

```bash
conda create -n covampnet python=3.10
conda activate covampnet
conda config --add channels conda-forge
conda install pyemma
pip install tensorflow==2.11.0
conda install h5py
conda install yaml
pip install typing-extensions
```

### Running the toy example

To try out VAMPnet with some very simple and small data, contained in this repository, run (with activated covampnet env):

```bash
python extract_mindist_representation.py --systems ZS-ab2 ZS-ab3 ZS-ab4
python compute_gradients.py --num_frames 5 --job_no 0 --systems ZS-ab2 ZS-ab3 ZS-ab4
python align_models.py --reference_system ZS-ab2 --other_systems ZS-ab3 ZS-ab4
python visualize_gradients.py --num_frames 5 --num_splits 1 --systems ZS-ab2 ZS-ab3 ZS-ab4 --reference_system ZS-ab2
```

### Reproducing CoVAMPnet results from the paper

1) Download data and replace the toy example data with the data from the paper:

```bash
wget https://data.ciirc.cvut.cz/public/projects/2023CoVAMPnet/covampnet_data.tar.gz
tar -xf covampnet_data.tar.gz
cp -TRv covampnet_data/ data/
mv covampnet_data/ data/
```

2) TODO:

### Using CoVAMPnet for your own data
Here we describe how to use our proposed directory structure (recommended). Alternatively, all the necessary filepaths can be edited in `config/paths.py`.

Prepare data:
1) Place trajectories inside `data/trajectories/SYSTEM_NAME/` - follow the diagram below (analogically to the toy example - data contained in this repo):
```
.
└── data/
    └── trajectories/
        ├── SYSTEM_NAME_1/
        │   ├── e1s1_0/
        │   │   └── trajectory_file.xtc
        │   ├── e1s2_0/
        │   │   └── trajectory_file.xtc
        │   ├── e*s*_0/  # "*" stands for integer number (episode and simulation ids respectively)
        │   │   └── trajectory_file.xtc
        │   ├── ...
        │   └── filtered.pdb # topology file for decoding of compressed .xtc trajectories
        ├── SYSTEM_NAME_2/
        │   └── ...
        └── ...
```

2) Place your models, files with validation loss scores and files with the precomputed inferred values for all simulation frames saved as `.hdf5` or `.p` (p for pickle) files (organized in the same way as the respective files obtained by the code in [Ab-42 Kinetic ensemble repo](https://github.com/vendruscolo-lab/ab42-kinetic-ensemble)) into their place in `data/`, see below:
```
.
└── data/
    ├── models/
    │   ├── SYSTEM_NAME_1/
    │   │   ├── model-ve-SYSTEM_NAME-M-MID-intermediate-2.hdf5 //Fill in name of your system, M= #markov_states , MID= model_id (i.e. 0-19 for 20 trained models for each system)
    │   │   └── ...
    │   └── SYSTEM_NAME_2/
    │       └── ...
    ├── model_loss_scores/
    │   ├── SYSTEM_NAME_1/
    │   │   ├── model-histories-SYSTEM_NAME-M-MID.p //Fill in name of your system, M= #markov_states , MID= model_id (i.e. 0-19 for 20 trained models for each system)
    │   │   └── ...    
    │   ├── SYSTEM_NAME_2/
    │   │   └── ...
    │   └── ...
    └── model_outputs/
        ├── SYSTEM_NAME_1/
        │   └── data.hdf5 //TODO
        ├── SYSTEM_NAME_2/
        │   └── ...
        └── ...

```

3) In terminal in the root directory, run the following sequence of commands (plugging in your system names, assuming the alignment should be performed w.r.t. SYSTEM_NAME_1):
```bash
python extract_mindist_representation.py --systems SYSTEM_NAME_1 SYSTEM_NAME_2 SYSTEM_NAME_3
python compute_gradients.py --num_frames 5 --job_no 0 --systems SYSTEM_NAME_1 SYSTEM_NAME_2 SYSTEM_NAME_3
python align_models.py --reference_system SYSTEM_NAME_1 --other_systems SYSTEM_NAME_2 SYSTEM_NAME_3
python visualize_gradients.py --num_frames 5 --num_splits 1 --systems SYSTEM_NAME_2 SYSTEM_NAME_3 --reference_system SYSTEM_NAME_1
```

TODO: iteration over the jobs

## Outputs of CoVAMPnet

Below we describe the use and interpretation of the outputs of CoVAMPnet.

### Alignment of Markov State Models for a single system and between two different systems

The sorters (files containing the indices, suggesting how to align the models both for a single system and between different systems) can be found in `results/sorters`.

The use of the sorters is demonstrated in the example below, we assume the toy example was run before to generate the toy data.
    
```python
import numpy as np
import h5py
from config.paths import MODEL_OUTPUTS_PATH_TEMPLATE
from config.data_model_params import NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, NUM_RESIDUES
from extract_mindist_representation import read_classification_scores, preprocess_trajectories
from visualize_gradients import read_sorters

#Prepara data for the example
SYSTEMS = ('ZS-ab2', 'ZS-ab3')
REF_SYSTEM = 'ZS-ab2'
frame_probs = [None, None]
for i,s in enumerate(SYSTEMS):
    data_path = MODEL_OUTPUTS_PATH_TEMPLATE.format(s)
    _, traj_lengths = preprocess_trajectories(s, nres=NUM_RESIDUES)
    total_frames_in_dataset = sum(traj_lengths)
    frame_probs[i] = read_classification_scores(s, NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, total_frames_in_dataset) #load Markov State probabilities for all frames and all models organized in a single array
frame_probs = tuple(frame_probs)

#Check data
print(f"frame_probs for {SYSTEMS[0]} shape = {frame_probs[0].shape}") #This should correspond to the shape of (NUM_MODELS, NUM_FRAMES, NUM_MARKOV_STATES)
print(f"frame_probs for {SYSTEMS[1]} shape = {frame_probs[1].shape}")

#Load precomputed alignments
local_sorters = {s: read_sorters(system=s, data='local') for s in SYSTEMS}
global_sorter = read_sorters(system=SYSTEMS[1], data='system', reference_system=REF_SYSTEM) #we are aligning wrt 'ZS-ab2', therefore alignment for 'ZS-ab2' is trivial and we dont need it

#Perform local alignments, to account for the arbitrary labeling each of the independently trained models have - even if all models are trained on the same data
aligned_probs_system_1 = frame_probs[0][:,:,local_sorters[SYSTEMS[0]]]
aligned_probs_system_2 = frame_probs[1][:,:,local_sorters[SYSTEMS[1]]]

#Perform global alignment of the second system wrt the first one
aligned_probs_system_2 = aligned_probs_system_2[:,:,global_sorter]

```

### Visualization of feature importance

The visualizations of the importance of particular inter-residue distances for the classification into particular Markov States can be found in `results/feature_importance`.

TODO: add figure
