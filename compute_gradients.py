"""

This script precomputes the gradients of the trained VAMPnet models with respect to the input simulation frames.
The script is supporting execution on HPC cluster, it can be run in parallel for example as an "array job" (as per
SLURM framework terminology) - to this end are presented two relevant arguments to this script (--num_frames and --job_no).
The argument --num_frames establishes the ammount of frames over which the gradients will be evaluated. The
--job_no argument serves as an ID which should be different for different jobs (even within the array). E.g. in the paper,
to evaluate the gradients over 10 000 frames we ran 2000 jobs, each computing gradients over 5 frames only - but with powerful
hardware (and large enough memory) it should be definitely possible to run less jobs with more frames per job.

Also to make sure that the sets of frames over which the gradients are evaluated in different jobs are disjoint,
the .yml file with the frame indices should be prepared apriori in results/frames_for_gradient_evaluation. If these .yaml
files are not present, they will be generated with random frames (not ensuring disjointness across jobs).

"""
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

from src.utils import unflatten
from os.path import join
import numpy as np
import yaml
import pickle
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import gc
import argparse
from src.model import KoopmanModel
from src.data_fcts import DataGenerator
from src.vanilla_gradients import VanillaGradients
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from config.paths import INFO_PATH_TEMPLATE, MODEL_PATH_TEMPLATE, MODEL_HISTORY_PATH_TEMPLATE, PRECOMPUTED_TRAJECTORIES_PATH_TEMPLATE, DATASPLITS_PATH_TEMPLATE, FRAMES_PER_JOBS_PATH_TEMPLATE, CLASSIFICATION_PER_JOBS_PATH_TEMPLATE, GRADIENTS_PER_JOBS_PATH_TEMPLATE
from config.data_model_params import MAX_FRAMES,SPLIT_RATIO,MD_TIME_RESOLUTION_IN_NS,NETWORK_LAG_IN_FRAMES,TRAININGS_PER_SPLIT, NUM_MARKOV_STATES, NUM_MODELS_PER_DATASET, NUM_TOTAL_TRAIN_ATTEMPTS
parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=5, help="number of frames to evaluate the gradients on")
parser.add_argument('--job_no', type=int, default=0, help='attempt number')
parser.add_argument('--systems', nargs='+',
                    help='List (separated by spaces) the names of the systems for which you wish to preprocess the data.',
                    required=True)

def select_frames_for_eval(args: argparse.Namespace, system: str, data_pars: dict) -> dict:
    """

    Parameters
    ----------
    args, argparse.Namespace - input by user
    system, str - name of the system at hand
    data_pars, DataParametes object

    Returns dict, read from or generated - containing the ids of the frames for classification and for evaluation of the gradients
    -------

    """
    file_path = FRAMES_PER_JOBS_PATH_TEMPLATE.format(d=system, nf=args.num_frames, jid=args.job_no)
    if os.path.exists(file_path):
        with open(file_path, 'r') as yamlfile:
            frame_ids = yaml.safe_load(yamlfile)
        print('Reading frames')
    else:
        with open(file_path, 'w') as outfile:
            frame_ids = {system: random.sample(range(0, data_pars[system].num_frames), args.num_frames)}  # Pick frames on which to evaluate for each system
            yaml.dump(frame_ids, outfile, default_flow_style=False)

    return frame_ids

class DataParameters:
    """
    Class reading and holding data and parameters for our analysis
    """

    def __init__(self, dataset_name, max_frames=MAX_FRAMES, ratio=SPLIT_RATIO, dt=MD_TIME_RESOLUTION_IN_NS, network_lag=NETWORK_LAG_IN_FRAMES, trainings_per_split=TRAININGS_PER_SPLIT, num_attempts=NUM_TOTAL_TRAIN_ATTEMPTS, num_markov_states=NUM_MARKOV_STATES):
        assert (NUM_TOTAL_TRAIN_ATTEMPTS//TRAININGS_PER_SPLIT)==NUM_MODELS_PER_DATASET, "Inconsistent number of total attempts, trainings per split and number of selected models"
        self.max_frames = max_frames
        self.num_markov_states = num_markov_states
        self.dataset_name = dataset_name
        self.ratio = ratio
        self.dt = dt
        self.network_lag = network_lag
        self.info_path = INFO_PATH_TEMPLATE.format(self.dataset_name)
        self.model_path = join(MODEL_PATH_TEMPLATE.format(d=self.dataset_name, total_states=self.num_markov_states))
        self.history_path = join(MODEL_HISTORY_PATH_TEMPLATE.format(d=self.dataset_name, total_states=self.num_markov_states))
        self.preprocessed_trajs_path = PRECOMPUTED_TRAJECTORIES_PATH_TEMPLATE.format(d=self.dataset_name)
        self.training_splits_path = DATASPLITS_PATH_TEMPLATE.format(d=self.dataset_name)

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
                with open(self.history_path.format(model_idx=model_idx), 'rb') as history_file:
                    pickle_object = pickle.load(history_file)
                losses = pickle_object['both'][0]['val_loss']
                min_loss = min(losses)
                subattempt_losses[subattempt] = min_loss
            vamp2_losses[out_attempt] = subattempt_losses
        vamp2_losses = np.array(vamp2_losses)
        return vamp2_losses

    def get_generator(self, attempt, multi_attempt_training=True) -> DataGenerator:
        """
        Loading the data generator given a particular data split

        Parameters
        ----------
        attempt, int - index of the datasplit for whit the generator should be constructed
        multi_attempt_training, bool - default = True, refers to the training scheme of more training performed with a single datasplit

        Returns the DataGenerator object
        -------

        """
        if multi_attempt_training:
            # THIS SHOULD ASSURE THAT THERE ARE THREE CONSECUTIVE ATTEMPTS ALL TRAINED ON SAME DATA SPLITS
            attempt = int(attempt / 3)
        else:
            raise ValueError('We should be working with the final version, relying on the multi-attempt training.')

        generator_path = self.training_splits_path.format(model_idx=attempt)
        generator = DataGenerator(self.input_data, ratio=self.ratio, dt=self.dt, max_frames=self.max_frames)
        generator.load(generator_path)
        return generator


    def load_koop(self, generator, attempt) -> KoopmanModel:
        """
        Loads a particular instance of a KoopmanModel containing the Artificial Neural Network we wish to analyze

        Parameters
        ----------
        generator, DataGenerator - used for the training of the given instance
        attempt, int - idx of the model instance to be loaded

        Returns, KoopmanModel
        -------

        """
        koop = KoopmanModel(n=self.num_markov_states, network_lag=self.network_lag, verbose=1)
        model_path = self.model_path.format(total_states=self.num_markov_states, model_idx=attempt)
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
            for old_object in ['koop', 'single_chi_model', 'generator']:
                if old_object in globals(): del old_object
            datasplit_id = selected_attempt//data_pars[system].num_selected_models #convert the overall train attempt into id of the data split
            generator = data_pars[system].get_generator(attempt=datasplit_id, multi_attempt_training=True)
            koop = data_pars[system].load_koop(generator=generator, attempt=selected_attempt)
            single_chi_model = tf.keras.models.Model(inputs=koop._models['chi']._model.input[0],
                                                     outputs=koop._models['chi']._model.layers[-2].output) #TODO: does it reinit the model correctly every time?
            single_chi_model.load_weights(filepath=data_pars[system].model_path.format(model_idx=selected_attempt), by_name=True)

            if tf.executing_eagerly():
                single_chi_model.compile(run_eagerly=True)
            else:
                raise ValueError("Eager execution is turned off! This wont allow us to evaluate the gradient values by Tensor conversion to numpy arrays.")
            single_models.append(single_chi_model)
            models_for_systems[system] = single_models


    v=VanillaGradients()
    CLASSES = (0,1,2)
    grads = {}
    classifications = {}

    start = time.time()
    for system in systems:
        logging.info(f'Iterating systems - system = {system}')
        frame_ids = select_frames_for_eval(args, system=system, data_pars=data_pars)
        params = data_pars[system]

        if os.path.exists(GRADIENTS_PER_JOBS_PATH_TEMPLATE.format(d=system,nf=args.num_frames,jid=args.job_no)) and os.path.exists('{}_classification_{}_job_{}.npy'.format(system,args.num_frames,args.job_no)):
            grads[system] = np.load(GRADIENTS_PER_JOBS_PATH_TEMPLATE.format(d=system,nf=args.num_frames,jid=args.job_no))
            classifications[system] = np.load(CLASSIFICATION_PER_JOBS_PATH_TEMPLATE.format(d=system,nf=args.num_frames,jid=args.job_no))
        elif (not os.path.exists(GRADIENTS_PER_JOBS_PATH_TEMPLATE.format(d=system,nf=args.num_frames,jid=args.job_no))) and (not os.path.exists('{}_classification_{}_job_{}.npy'.format(system,args.num_frames,args.job_no))):
            grads_per_system = [[[0]*args.num_frames for _ in range(NUM_MARKOV_STATES)] for _ in range(params.num_selected_models)] # np.zeros((20,3,FRAMES_PER_SYSTEM,1,780))
            class_scores_per_system = np.zeros((params.num_selected_models,NUM_MARKOV_STATES,args.num_frames))

            for m_idx, m in enumerate(models_for_systems[system]):
                logging.info('Model {} evaluation'.format(m_idx))
                for c_idx,c in enumerate(CLASSES): #for alignment plug in proper sorter instead of CLASSES
                    for frame_idx, frame_id in enumerate(frame_ids[system]): #TODO: check the alignment + parallelize
                        frame = koop.generator.data_flat[frame_id].reshape(1,-1)
                        grads_per_system[m_idx][c_idx][frame_idx] = v.explain(validation_data=frame, model=m, class_index=c)[0,:] #here potential alignment from c to c_idx
                        class_scores_per_system[m_idx, c_idx, frame_idx] = m.predict(frame)[0][c] #here potential alignment conversion from c to c_idx
                del m
                gc.collect()
            grads[system] = np.array(grads_per_system)
            classifications[system] = class_scores_per_system

            np.save(GRADIENTS_PER_JOBS_PATH_TEMPLATE.format(d=system, nf=args.num_frames, jid=args.job_no),grads[system])
            np.save(CLASSIFICATION_PER_JOBS_PATH_TEMPLATE.format(d=system, nf=args.num_frames, jid=args.job_no),classifications[system])
        else:
            raise ValueError('Only one of the files (grads or classifications) is missing. Either both or none of them should be precomputed - to ensure consistency.')

        logging.info(f"Classifications for frames specified in {FRAMES_PER_JOBS_PATH_TEMPLATE.format(d= system, nf=args.num_frames, jid=args.job_no)} and for the system {system} computed and saved to {CLASSIFICATION_PER_JOBS_PATH_TEMPLATE.format(d=system,nf=args.num_frames,jid=args.job_no)}")
        logging.info(f"Gradients for system {system} computed and saved to {GRADIENTS_PER_JOBS_PATH_TEMPLATE.format(d=system, nf=args.num_frames, jid=args.job_no)}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.systems)
