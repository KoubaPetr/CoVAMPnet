from src.utils import unflatten
from os.path import join
import numpy as np
import yaml
import pickle
import random
import os
import time
import gc
import argparse
from src.model import KoopmanModel
from src.data_fcts import DataGenerator
from src.vanilla_gradients import VanillaGradients
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=5, help="number of frames to evaluate the gradients on")
parser.add_argument('--job_no', type=int, default=0, help='attempt number')
args = parser.parse_args()


class DataParameters:
    """
    Class reading and holding data and parameters for our analysis
    """

    def __init__(self, dataset_name, max_frames=650000, ratio=0.9, dt=0.1, network_lag=50, trainings_per_split=3, num_attempts=6):
        self.max_frames = max_frames
        self.dataset_name = dataset_name
        self.ratio = ratio
        self.dt = dt
        self.network_lag = network_lag
        self.info_path = 'data/trajectories/{}/info.yml'.format(self.dataset_name)
        self.model_path = join("data/models/{}".format(self.dataset_name), 'model-ve-{}'.format(self.dataset_name) + '-{}-{}-intermediate-2.hdf5')
        self.history_path = join("data/model_scores/{}".format(self.dataset_name), 'model-histories-{}'.format(self.dataset_name) + '-{}-{}.p')
        self.preprocessed_trajs_path = "data/precomputed_trajectories/mindist-780-{}.npy".format(self.dataset_name)
        self.training_splits_path = 'data/training_splits/{}'.format(self.dataset_name)

        self.trainings_per_split = trainings_per_split
        self.num_selected_models = num_attempts//self.trainings_per_split

        assert num_attempts%self.trainings_per_split == 0, f" Total number of training attempts {num_attempts} should be divisible by the number of attempts per datasplit {self.trainings_per_split}"

        with open(self.info_path, "r") as yaml_file:
            self.lengths = yaml.safe_load(yaml_file)['lengths']
            self.num_frames = sum(self.lengths)

        self.input_data = self.load_input_data()
        self.VAMP_scores_for_attempts = self.read_VAMP_scores_for_attempts()  # for negative vamp score such as is the case
        self.selected_models = [trainings_per_split * row_id + np.argmin(row) for row_id, row in
                                enumerate(self.VAMP_scores_for_attempts)]

    def load_input_data(self) -> np.ndarray:
        """
        Load the preprocessed trajectories
        """
        raw = np.load(self.preprocessed_trajs_path)
        raw_mean, raw_std = raw.mean(axis=0), raw.std(axis=0)
        input_data = [(r - raw_mean) / raw_std for r in unflatten(raw, [self.lengths])]
        return input_data

    def read_VAMP_scores_for_attempts(self) -> np.ndarray:
        """
        Reads in the logged validation loss, which is used to select the best of the training attempts performed on the
        same datasplit

        Returns np.ndarray of shape (num_selected_models, trainings_per_split), containing the final val losses for
        given training attempts.
        -------

        """
        vamp2_losses = [0] * self.num_selected_models

        for out_attempt in range(self.num_selected_models):
            subattempt_losses = np.zeros(self.trainings_per_split)
            for subattempt in range(self.trainings_per_split):
                model_idx = self.trainings_per_split * out_attempt + subattempt
                with open(self.history_path.format(3, model_idx), 'rb') as history_file:
                    pickle_object = pickle.load(history_file)
                losses = pickle_object['both'][0]['val_loss']
                min_loss = min(losses)
                subattempt_losses[subattempt] = min_loss
            vamp2_losses[out_attempt] = subattempt_losses
        vamp2_losses = np.array(vamp2_losses)
        return vamp2_losses

    def get_generator(self, attempt, multi_attempt_training=True):
        if multi_attempt_training:
            # THIS SHOULD ASSURE THAT THERE ARE THREE CONSECUTIVE ATTEMPTS ALL TRAINED ON SAME DATA SPLITS
            attempt = int(attempt / 3)
        else:
            raise ValueError('We should be working with the final version, relying on the multi-attempt training.')

        generator_path = join(self.training_splits_path, "model-idx-{0}-{1}.hdf5".format(self.dataset_name, attempt))
        generator = DataGenerator(self.input_data, ratio=self.ratio, dt=self.dt, max_frames=self.max_frames)
        generator.load(generator_path)
        return generator


    def load_koop(self, generator, attempt, n=3):
        koop = KoopmanModel(n=n, network_lag=self.network_lag, verbose=1)
        model_path = self.model_path.format(n, attempt)
        koop.load(model_path)
        koop.generator = generator
        return koop

    def __repr__(self):
        return "DataParameters object for system {}".format(self.dataset_name)


def main(systems: list[str] = ['ZS-ab2', 'ZS-ab3', 'ZS-ab4']):

    data_pars = {system: DataParameters(dataset_name=system) for system in systems}

    models_for_systems = {}
    for system in systems:
        single_models = []
        for selected_attempt in data_pars[system].selected_models:
            if 'koop' in globals():
                del koop
            if 'single_chi_model' in globals():
                del single_chi_model
            if 'generator' in globals():
                del generator
            print(selected_attempt)
            attempt_20 = selected_attempt//20
            generator = data_pars[system].get_generator(attempt=attempt_20, multi_attempt_training=True)
            koop = data_pars[system].load_koop(generator=generator, attempt=selected_attempt, n=3)
            single_chi_model = tf.keras.models.Model(inputs=koop._models['chi']._model.input[0],
                                                     outputs=koop._models['chi']._model.layers[-2].output) #TODO: does it reinit the model correctly every time?
            single_chi_model.load_weights(filepath=data_pars[system].model_path.format(3, selected_attempt), by_name=True)
            #koop._models["single_chi"] = single_chi_model
            if tf.executing_eagerly():
                single_chi_model.compile(run_eagerly=True)
            else:
                raise ValueError("Eager execution is turned off! This wont allow us to evaluate the gradient values by Tensor conversion to numpy arrays.")
            single_models.append(single_chi_model)
            models_for_systems[system] = single_models

    FRAMES_PER_SYSTEM = args.num_frames
    job_no = args.job_no

    if os.path.exists('{}_frames_for_gradient_evaluation_job_{}.yml'.format(FRAMES_PER_SYSTEM, job_no)):
        with open('{}_frames_for_gradient_evaluation_job_{}.yml'.format(FRAMES_PER_SYSTEM, job_no), 'r') as yamlfile:
            FRAME_IDs = yaml.safe_load(yamlfile)
        print('Reading frames')
    else:
        with open('{}_frames_for_gradient_evaluation_job_{}.yml'.format(FRAMES_PER_SYSTEM,job_no), 'w') as outfile:
            FRAME_IDs = {system: random.sample(range(0, data_pars[system].num_frames), FRAMES_PER_SYSTEM) for system in systems} # Pick frames on which to evaluate for each system
            yaml.dump(FRAME_IDs, outfile, default_flow_style=False)


    v=VanillaGradients()
    CLASSES = (0,1,2)
    grads = {}
    classifications = {}

    start = time.time()
    for system in systems: #TODO redesign the loop
        print('Iterating systems')
        params = data_pars[system]
        if os.path.exists('{}_grads_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no)) and os.path.exists('{}_classification_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no)):
            grads[system] = np.load('{}_grads_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no))
            classifications[system] = np.load('{}_classification_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no))
        elif (not os.path.exists('{}_grads_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no))) and (not os.path.exists('{}_classification_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no))):
            grads_per_system = [[[0]*FRAMES_PER_SYSTEM for _ in range(3)] for _ in range(params.num_selected_models)] # np.zeros((20,3,FRAMES_PER_SYSTEM,1,780))
            class_scores_per_system = np.zeros((params.num_selected_models,3,FRAMES_PER_SYSTEM))

            for m_idx, m in enumerate(models_for_systems[system]):
                print('model {} evaluation'.format(m_idx))
                for c_idx,c in enumerate(CLASSES): #for alignment plug in proper sorter instead of CLASSES
                    for frame_idx, frame_id in enumerate(FRAME_IDs[system]): #TODO: check the alignment + parallelize
                        frame = koop.generator.data_flat[frame_id].reshape(1,-1)
                        grads_per_system[m_idx][c_idx][frame_idx] = v.explain(validation_data=frame, model=m, class_index=c)[0,:] #here potential alignment from c to c_idx
                        class_scores_per_system[m_idx, c_idx, frame_idx] = m.predict(frame)[0][c] #here potential alignment conversion from c to c_idx
                del m
                gc.collect()
            grads[system] = np.array(grads_per_system)
            classifications[system] = class_scores_per_system

            np.save('{}_grads_{}_job_{}.npy'.format(system, FRAMES_PER_SYSTEM,job_no),grads[system])
            np.save('{}_classification_{}_job_{}.npy'.format(system,FRAMES_PER_SYSTEM,job_no),classifications[system])
        else:
            raise ValueError('Only one of the files (grads or classifications) is missing')

if __name__ == '__main__':
    main()
