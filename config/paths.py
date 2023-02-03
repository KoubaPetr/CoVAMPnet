TRAJECTORY_PATHS_TEMPLATE = "data/trajectories/{}/e*s*_*/output.filtered.xtc"
TOPOLOGY_PATH_TEMPLATE = "data/trajectories/{}/filtered.pdb"
MINDIST_PATH_TEMPLATE = 'data/precomputed_trajectories/mindist-780-{}.npy'
INFO_PATH_TEMPLATE = 'data/trajectories/{}/info.yml'
MODEL_PATH_TEMPLATE = "data/models/{d}/model-ve-{d}-{total_states}-{{model_idx}}-intermediate-2.hdf5"
MODEL_HISTORY_PATH_TEMPLATE = "data/model_scores/{d}/model-histories-{d}-{total_states}-{{model_idx}}.p"
PRECOMPUTED_TRAJECTORIES_PATH_TEMPLATE = "data/precomputed_trajectories/mindist-780-{d}.npy"
DATASPLITS_PATH_TEMPLATE = 'data/training_splits/{d}/model-idx-{d}-{{model_idx}}.hdf5'
MODEL_OUTPUTS_PATH_TEMPLATE = 'data/model_outputs/{d}/data.hdf5'

FRAMES_PER_JOBS_PATH_TEMPLATE = 'results/frames_for_gradient_jobs/{}_grads_{}_job_{}.npy'
GRADIENTS_PER_JOBS_PATH_TEMPLATE = 'results/gradients/{}_grads_{}_job_{}.npy'
CLASSIFICATION_PER_JOBS_PATH_TEMPLATE = 'results/classification/{}_classification_{}_job_{}.npy'
CLUSTER_AVG_PATH_TEMPLATE = 'results/classification/{d}_cluster_avg_mindist_{ms}_markov_states.npy'

LOCAL_SORTERS_PATH_TEMPLATE = 'results/sorters/local_sorters_{d}.yml'
SYSTEM_SORTERS_PATH_TEMPLATE = 'results/sorters/system_sorters_{d}_wrt_{ref_data}.yml'

FEATURE_IMPORTANCE_PATH_TEMPLATE = 'results/feature_importance/{}_feature_importance.png'
FEATURE_IMPORTANCE_FULL_MATRIX_PATH_TEMPLATE = 'results/feature_importance/{}_feature_importance_full_matrix.png'