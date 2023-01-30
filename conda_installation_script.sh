conda create -n covampnet python=3.10
conda activate covampnet
conda config --add channels conda-forge
conda install pyemma
pip install tensorflow==2.11.0
conda install h5py
conda install yaml