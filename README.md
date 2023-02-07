# Comparative Markov State Analysis (CoVAMPnet)

Code supplementing the paper "Effects of Alzheimer’s Disease Drug Candidates on Disordered Aβ42 Dissected by Comparative Markov State Analysis CoVAMPnet" [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.01.06.523007v1)

For training the VAMPnet and for performing the analyses over a single system, we acknowledge and refer to the [repository](https://github.com/vendruscolo-lab/ab42-kinetic-ensemble).

For the Comparative Markov State Analysis (CoVAMPnet), we provide the code in this repository and supplementing data [Here (17 GB)](https://data.ciirc.cvut.cz/public/projects/2023CoVAMPnet/covampnet_data.tar.gz).

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

### Using CoVAMPnet for your own data

1) Prepare data: TODO
Put trajectories inside data/trajectories/SYSTEM_NAME/ - as in the toy example (i.e. .xtc data inside e*s*_0/ folders and filtered.pdb topology file placed next to the e*s*_0/ folders)
3) Run python extract_mindist_representation.py --systems ZS-ab2 ZS-ab3 ZS-ab4
4) Run python compute_gradients.py --num_frames 5 --job_no 0 --systems ZS-ab2 ZS-ab3 ZS-ab4 #(just as a test, for real reproduction, first place .yml files listing the frames for each job into results/frames_for_gradient_jobs and run the jobs, probably using an HPC scheduler system)
5) Run python align_models.py --reference_system ZS-ab2 --other_systems ZS-ab3 ZS-ab4
6) Run python visualize_gradients.py --num_frames 5 --num_splits 1 --systems ZS-ab2 ZS-ab3 ZS-ab4 --reference_system ZS-ab2

## Outputs of CoVAMPnet

Below we describe the use and interpretation of the outputs of CoVAMPnet.

### Alignment of Markov State Models for a single system and between two different systems

The sorters (files containing the indices, suggesting how to align the models both for a single system and between different systems) can be found in `results/sorters`.

The use of the sorters is the following:
    
TODO

### Visualization of feature importance

The visualizations of the importance of particular inter-residue distances for the classification into particular Markov States can be found in `results/feature_importance`.
