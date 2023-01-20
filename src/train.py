import gc
import h5py
from datetime import datetime
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
import random as python_random
from glob import glob
import pyemma as pe

from model import unflatten, DataGenerator, KoopmanModel
from utils import (get_new_outsizes, get_last_step_model_path, Quantities,
                   system_names, Utils)


class Logger:
    def __init__(self, logfile):
        self.logfile = logfile

    def __call__(self, text):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.logfile, 'a') as f:
            f.write('\n[{0}] {1}'.format(current_time, text))
        print(text)

class IntermediateSaveCallback:
    def __init__(self, run_path, k, n, attempt, logger=None):
        self.run_path = run_path
        self.k = k
        self.n = n
        self.attempt = attempt
        self.num_calls = 0
        self.logger = print if logger is None else logger

    def __call__(self, koop):
        self.logger("Saving intermediate model ({0})".format(self.num_calls))

        model_path = join(self.run_path,
                          "training/models/model-ve-{0}-{1}-{2}-intermediate-{3}.hdf5".format(self.k, self.n,
                                                                                              self.attempt,
                                                                                              self.num_calls))
        koop.save(model_path)
        self.num_calls += 1


def get_histories(koop):
    histories = {}
    for name, model in koop._models.items():
        if model._history:
            histories[name] = []
            for history in model._history:
                if history is None:
                    histories[name].append(None)
                else:
                    histories[name].append(history.history)
    return histories


# def show_histories(histories):
#     for model, model_histories in histories.items():
#         print(model)
#         for i, history in enumerate(model_histories):
#             if history is not None:
#                 if 'metric_VAMP' in history:
#                     for key in ['metric_VAMP', 'val_metric_VAMP']:
#                         plt.plot(history[key], label=key)
#                     plt.title('VAMP, training {0}'.format(i+1))
#                     plt.legend()
#                     plt.show()
#                 for key in ['loss', 'val_loss']:
#                     plt.plot(history[key], label=key)
#                 plt.title('Loss, training {0}'.format(i+1))
#                 plt.legend()
#                 plt.show()

def save_histories(filename, koop):
    histories = get_histories(koop)
    with open(filename, 'wb') as f:
        pickle.dump(histories, f, pickle.HIGHEST_PROTOCOL)


def set_seed(seed=None):
    if seed is None:
        seed = python_random.randrange(2 ** 32 - 1)
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    return seed


class Trainer(Utils):
    def __init__(self, base_run_path, k, verbose=True, epoch_id = None):
        self.base_run_path = base_run_path
        self.k = k
        self.verbose = verbose

        if 'SM' in self.k:
            _md_data = 'sergio'
        elif 'ZS' in self.k:
            _md_data = 'zainab'

        self.quantities = Quantities(self.base_run_path, self.k, md_data=_md_data,
                                     verbose=self.verbose, train=True, epoch_id=epoch_id)

        # Check that lag times won't exceed minimum trajectory length
        # trajectory_length = min(array.min() for array in self.lengths)
        # assert max(self.lags) < trajectory_length
        # assert self.network_lag * (self.steps - 1) < trajectory_length

        os.makedirs(join(self.training_path, "models"), exist_ok=True)
        os.makedirs(join(self.training_path, "logs"), exist_ok=True)
        os.makedirs(join(self.run_path, "results"), exist_ok=True)

        self._print("Trajectories: {0}".format(len(self.trajs)))
        self._print("Frames: {0}".format(self.nframes))
        self._print("Time: {0:5.3f} mu_s".format(self.nframes * self.dt / 1000))

    def __getattr__(self, attr):
        return getattr(self.quantities, attr)

    def get_generator(self, logger, attempt, multi_attempt_training = False):

        if multi_attempt_training:
            attempt = int(attempt/3) #THIS SHOULD ASSURE THAT THERE ARE THREE CONSECUTIVE ATTEMPTS ALL TRAINED ON SAME DATA SPLITS

        generator = DataGenerator(self.input_data, ratio=self.ratio, dt=self.dt,
                                  max_frames=self.max_frames)
        generator_path = join(self.training_path, "models/model-idx-{0}-{1}.hdf5".format(self.k, attempt))
        if os.path.isfile(generator_path):
            logger("Found existing generator.")
            generator.load(generator_path)
        else:
            logger("Saving new generator.")
            generator.save(generator_path)
        return generator

    def train_koop(self, logger, generator, attempt, n):
        seed_path = join(self.training_path, "models/model-seed-{0}-{1}-{2}.txt".format(self.k, n, attempt))
        if os.path.isfile(seed_path):
            logger("Found existing seed.")
            seed = int(np.loadtxt(seed_path, dtype=int))
            new_seed = set_seed(seed)
            assert seed == new_seed
        else:
            logger("Generating new seed.")
            seed = set_seed()
            with open(seed_path, "w") as file:
                file.write(str(seed))

        callback = IntermediateSaveCallback(self.run_path, self.k, n, attempt,
                                            logger=logger)
        koop = KoopmanModel(callbacks=[callback], n=n,
                            network_lag=self.network_lag, verbose=1,
                            nnargs=self.nnargs)

        logger("Training {0} n={1} i={2} with seed {3}...".format(self.k, n, attempt, seed))
        koop.fit(generator)

        # Saving the loss history
        save_histories(join(self.training_path, "models/model-histories-{0}-{1}-{2}.p".format(self.k, n, attempt)),
                       koop)
        return koop

    def load_koop(self, logger, generator, attempt, n):
        logger("Loading Koopman operator from file (k={0} n={1} i={2})".format(self.k, n, attempt))
        koop = KoopmanModel(n=n, network_lag=self.network_lag, verbose=1,
                            nnargs=self.nnargs)
        model_path = get_last_step_model_path(self.run_path, self.k, n, attempt)
        koop.load(model_path)
        koop.generator = generator
        return koop

    def save_results(self, logger, generator, attempt, n, koop, cktest_only = False, epoch_id = None): #TODO: based on epoch_id is/isnt None do different kinds of computations (dont do /do CK tests etc.)
        data_path = join(self.run_path, "results/data-{0}.hdf5".format(attempt))
        with h5py.File(data_path, "a") as write:
            # Create HDF5 groups
            ens = write.require_group(self.k)
            att = ens.require_group(str(attempt))
            out = att.require_group(str(n))
            if epoch_id is not None: #in case of convergence training only
                logger("Estimating implied timescales...")
                its = out.require_dataset("its", shape=(n - 1, len(self.lags)), dtype="float32")
                its[:] = koop.its(self.lags)
            else:
                if not cktest_only:
                    logger("Estimating intermediate Koopman operator at lag {0}...".format(self.analysis_lag))
                    ko = out.require_dataset("k", shape=(n, n), dtype="float32")
                    ko[:] = koop.estimate_koopman(lag=self.analysis_lag)
                    logger("Estimating mu...")
                    mu = out.require_dataset("mu", shape=(koop.data.n_train,), dtype="float32")
                    mu[:] = koop.mu
                    logger("Estimating implied timescales...")
                    its = out.require_dataset("its", shape=(n - 1, len(self.lags)), dtype="float32")
                    its[:] = koop.its(self.lags)
                    logger("Performing CK-test...")
                    cke = out.require_dataset("cke", shape=(n, n, self.steps), dtype="float32")
                    ckp = out.require_dataset("ckp", shape=(n, n, self.steps), dtype="float32")

                    from utils import compute_lag
                    analysis_lag = compute_lag(self.cfg["analysis"]["lag_ns"], self.dt)
                    logger("analysis_lag={}".format(analysis_lag))
                    cke[:], ckp[:] = koop.cktest(self.steps, analysis_lag)
                    logger("Estimating chi...")
                    bootstrap = out.require_dataset("bootstrap", shape=(koop.data.n_train, 2 * n), dtype="float32")
                    bootstrap[:] = koop.transform(koop.data.trains[0])
                    full = out.require_dataset("full", shape=(self.nframes, 2 * n), dtype="float32")
                    full[:] = koop.transform(generator.data_flat)
                else:
                    logger("Performing CK-test...")
                    cke = out.require_dataset("cke", shape=(n, n, self.steps), dtype="float32")
                    ckp = out.require_dataset("ckp", shape=(n, n, self.steps), dtype="float32")

                    from utils import compute_lag
                    analysis_lag = compute_lag(self.cfg["analysis"]["lag_ns"], self.dt)
                    logger("analysis_lag={}".format(analysis_lag))
                    cke[:], ckp[:] = koop.cktest(self.steps, analysis_lag)

    def __call__(self, attempt, outsizes=None, train=True, results=True, cktest_only=False, multi_attempt_training=False, epoch_id=None):
        if outsizes is None:
            outsizes = get_new_outsizes(self.run_path, self.k)
            outsizes = np.array(outsizes).reshape(-1)  # ensure 1D array
        self._print("Outsizes:", outsizes)

        logger = Logger(join(self.training_path, "logs/training-log-{0}-{1}.txt".format(self.k, attempt)))
        logger("Starting logs.")
        generator = self.get_generator(logger, attempt, multi_attempt_training=multi_attempt_training)

        for n in outsizes:
            try:
                if train:
                    koop = self.train_koop(logger, generator, attempt, n)
                else:
                    koop = self.load_koop(logger, generator, attempt, n)

                if results:
                    self.save_results(logger, generator, attempt, n, koop, cktest_only=cktest_only, epoch_id=epoch_id)

                logger("Finished this attempt.")
            except Exception as e:
                logger("Error: {0}".format(e))
            finally:
                del koop
                gc.collect()
        logger("Done.")


def main():
    assert len(tf.config.list_physical_devices("GPU")) > 0, "No GPU found."

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", metavar="path", type=str,
                        help="path to the run directory")
    parser.add_argument("system", metavar="idx", type=int,
                        help="index of chosen system")
    parser.add_argument('attempt', metavar='attempt', type=int,
                        help='attempt number')
    parser.add_argument("-v", "--verbose", help="verbose",
                        action="store_true")
    parser.add_argument("-L", "--load", help="load koopman, no training",
                        action="store_true")
    parser.add_argument("--ck_only", help="recompute only CK test",
                        action="store_true")
    parser.add_argument("--multi_attempt_training", help="in case of training with 3 consecutive attempts trained on same data split",
                        action="store_true")
    parser.add_argument("--convergence_training", help="in case of training on progressively growing dataset to observe the convergence of implied timescales wrt dataset size",
                        action="store_true")
    args = parser.parse_args()
    print(args)

    system_idx = args.system
    base_run_path = args.run_path
    attempt = args.attempt
    verbose = args.verbose
    train = not args.load
    k = system_names[system_idx]

    if not args.convergence_training: #usual training
        trainer = Trainer(base_run_path, k, verbose=verbose)
        trainer(attempt, train=train, cktest_only=args.ck_only, multi_attempt_training=args.multi_attempt_training)
    else: #training to get evaluate the convergence wrt dataset size
        for epoch_id in range(1,17):
            trainer = Trainer(base_run_path, k, verbose=verbose, epoch_id=epoch_id)
            trainer(attempt, train=train, cktest_only=args.ck_only, multi_attempt_training=args.multi_attempt_training, epoch_id=epoch_id)



if __name__ == "__main__":
    main()
