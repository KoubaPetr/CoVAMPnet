1) Put trajectories inside data/trajectories/SYSTEM_NAME/ - as in the example (i.e. .xtc data inside e*s*_0/ folders and filtered.pdb topology file placed next to the e*s*_0/ folders) #TODO: provide the option to download the Zenodo data
2) Run python extract_mindist_representation.py --systems ZS-ab2 ZS-ab3 ZS-ab4
3) Run python compute_gradients.py --num_frames 5 --job_no 0 --systems ZS-ab2 ZS-ab3 ZS-ab4 #(just as a test, for real reproduction, first place .yml files listing the frames for each job into results/frames_for_gradient_jobs and run the jobs, probably using an HPC scheduler system)
4) Run python align_models.py --reference_system ZS-ab2 --other_systems ZS-ab3 ZS-ab4
5) Run python visualize_gradients.py --num_frames 5 --num_splits 1 --systems ZS-ab2 ZS-ab3 ZS-ab4
