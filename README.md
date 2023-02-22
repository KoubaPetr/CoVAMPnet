# Comparative Markov State Analysis (CoVAMPnet)

Code supplementing the paper "Effects of Alzheimer’s Disease Drug Candidates on Disordered Aβ42 Dissected by Comparative Markov State Analysis CoVAMPnet" [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.01.06.523007v1)

For training the VAMPnet and for performing the analyses over a single system, we acknowledge and refer to the [Ab-42 Kinetic ensemble repo](https://github.com/vendruscolo-lab/ab42-kinetic-ensemble).

For the Comparative Markov State Analysis (CoVAMPnet), we provide the code in this repository and supplementing data [on project web page (17 GB)](https://data.ciirc.cvut.cz/public/projects/2023CoVAMPnet/covampnet_data.tar.gz).

## Citation
If you use this code in your research resulting in an academic publication, please cite our work by using the following citation:
```
@article{covampnet2023biorxiv,
    title={Effects of Alzheimer’s Disease Drug Candidates on Disordered Aβ42 Dissected by Comparative Markov State Analysis CoVAMPnet},
    author={Sérgio M. Marques and Petr Kouba and Anthony Legrand and Jiri Sedlar and Lucas Disson and Joan Planas-Iglesias and Zainab Sanusi and Antonin Kunka and Jiri Damborsky and Tomas Pajdla and Zbynek Prokop and Stanislav Mazurenko and Josef Sivic and David Bednar},
    journal={bioRxiv preprint bioRxiv:2023.01.06.523007},
    year={2023}
}
```
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
cp -TRv CoVAMPnet_data/ data/
rm -rf CoVAMPnet_data/
```

2) Preprocess data:

```bash
python extract_mindist_representation.py --systems ZS-ab2 ZS-ab3 ZS-ab4
```

3) Precompute gradients. This part can be easily parallelized and typically this would be done on an HPC cluster. Therefore, the particular commands would depend on users cluster and its scheduling system. We used cluster powered by SLURM scheduling system and the command there would be the following. The user is advised to modify the comand and the corresponding job script `gradient_job.sh`.

```bash
sbatch --array=0-1999 gradient_job.py
```

Alternatively (might be very slow, some parallelization is advised), the following command can be run:

```bash
counter=0
while [ $counter -le 2000 ]
do
echo $counter
python gradient_job.py --num_frames 5 --job_no $counter
((counter++))
done
```

4) Align the models and visualize the gradients:

```bash
python align_models.py --reference_system ZS-ab2 --other_systems ZS-ab3 ZS-ab4
python visualize_gradients.py --num_frames 5 --num_splits 2000 --systems ZS-ab2 ZS-ab3 ZS-ab4 --reference_system ZS-ab2
```
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
    │   │   ├── model-ve-SYSTEM_NAME-M-MID-intermediate-2.hdf5 #Fill in name of your system, M= #markov_states , MID= model_id (i.e. 0-19 for 20 trained models for each system)
    │   │   └── ...
    │   └── SYSTEM_NAME_2/
    │       └── ...
    ├── model_loss_scores/
    │   ├── SYSTEM_NAME_1/
    │   │   ├── model-histories-SYSTEM_NAME-M-MID.p #Fill in name of your system, M= #markov_states , MID= model_id (i.e. 0-19 for 20 trained models for each system)
    │   │   └── ...    
    │   ├── SYSTEM_NAME_2/
    │   │   └── ...
    │   └── ...
    └── model_outputs/
        ├── SYSTEM_NAME_1/
        │   └── data.hdf5 #TODO
        ├── SYSTEM_NAME_2/
        │   └── ...
        └── ...

```

3) In terminal in the root directory, run the following command to preprocess your data (plugging in your system names, example: `SYSTEM_NAME_1=ZS-ab2`):
```bash
SYSTEM_NAME_1=...
SYSTEM_NAME_2=...
SYSTEM_NAME_3=...

python extract_mindist_representation.py --systems $SYSTEM_NAME_1 $SYSTEM_NAME_2 $SYSTEM_NAME_3
```

4) Before precomputing the gradients, decide over how many MD frames the gradients should be evaluated for each system. For computational efficiency it is recommended to parallelize the gradient computation. To this end, specify 2 parameters NUM_FRAMES_PER_JOB and NUM_JOBS, where the total number of frames used for evaluation will be the product of these two parameters. Choose the balance of the two parameters, according to what can be handled by users machine memory. Optionally (recommended), the selection of frames for the gradient computation can be precomputed, to ensure each frame is drawn only at maximum once across all jobs. This precomputation can be achieved by calling the following script:

```bash
NUM_FRAMES_PER_JOB=...
NUM_JOBS=...
python select_frames_for_gradient_evaluation --frames_per_split $NUM_FRAMES_PER_JOB --num_splits $NUM_JOBS
```

5) Similarly to the paragraph on reproduction of paper values, the precise way of parallelization will depend on your machine. Below is an example of using SLURM, but the corresponding script needs to be adjusted to match users cluster. Alternatively

```bash
sbatch --array=0-$(($NUM_JOBS-1)) gradient_job.py
```

Alternatively (might be very slow, some parallelization is advised), the following command can be run:

```bash
counter=0
while [ $counter -le $NUM_JOBS ]
do
echo $counter
python gradient_job.py --num_frames $NUM_FRAMES_PER_JOB --job_no $counter
((counter++))
done
```

6) Align the models and visualize the gradients, to this end reference system with respect to which the alignment should be performed, needs to be specified:
```bash
python align_models.py --reference_system $SYSTEM_NAME_1 --other_systems $SYSTEM_NAME_2 $SYSTEM_NAME_3
python visualize_gradients.py --num_frames $NUM_FRAMES_PER_JOB --num_splits $NUM_JOBS --systems $SYSTEM_NAME_1 $SYSTEM_NAME_2 $SYSTEM_NAME_3 --reference_system $SYSTEM_NAME_1
```

## Outputs of CoVAMPnet

Below we describe the use and interpretation of the outputs of CoVAMPnet.

### Alignment of Markov State Models for a single system and between two different systems

The sorters (files containing the indices, suggesting how to align the models both for a single system and between different systems) can be found in `results/sorters`.

The use of the sorters is demonstrated in the example below, we assume the toy example was run before to generate the toy data.
    
```python
import numpy as np

from config.paths import MODEL_OUTPUTS_PATH_TEMPLATE
from config.data_model_params import NUM_MODELS_PER_DATASET, NUM_MARKOV_STATES, NUM_RESIDUES
from extract_mindist_representation import read_classification_scores, preprocess_trajectories
from visualize_gradients import read_sorters

#Prepara data for the example
SYSTEMS = ('ZS-ab2', 'ZS-ab4')
REF_SYSTEM = 'ZS-ab2'
frame_probs = [None, None]
for i,s in enumerate(SYSTEMS):
    data_path = MODEL_OUTPUTS_PATH_TEMPLATE.format(d=s)
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
aligned_probs_system_1 = np.array([frame_probs[0][i,:,local_sorters[SYSTEMS[0]][NUM_MARKOV_STATES][i]] for i in range(NUM_MODELS_PER_DATASET)]).transpose(0,2,1)
aligned_probs_system_2 = np.array([frame_probs[1][i,:,local_sorters[SYSTEMS[1]][NUM_MARKOV_STATES][i]] for i in range(NUM_MODELS_PER_DATASET)]).transpose(0,2,1)

#Perform global alignment of the second system wrt the first one
aligned_probs_system_2 = aligned_probs_system_2[:,:,global_sorter[NUM_MARKOV_STATES]]

```

### Visualization of feature importance

The visualizations of the importance of particular inter-residue distances for the classification into particular Markov States can be found in `results/feature_importance`. Example of such a visualization is presented in **Fig.1** below. Based on the red and blue pixels in these visualizations, it can be inferred which pairs of residues are preferred to be far from (or close to) one another by which Markov state.

| ![Alt text](results/examples/ZS-ab2_saliency_full.png) | 
|:--| 
| **Fig. 1:** *Visualization of the feature importance for assignment of inter-residue distance maps representing MD frames into particular Markov state. Row number in the grid denotes Markov state with respect to which the assignment probability is changing based on an increase in distance of two residues denoted by the coordinates in the subfigure. The column number denotes the Markov state to which the MD frames used for the evaluation belong.* |
