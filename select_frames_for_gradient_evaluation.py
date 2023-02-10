import random
import argparse
import yaml
from compute_gradients import DataParameters
from tqdm import trange
from config.paths import FRAMES_PER_JOBS_PATH_TEMPLATE

parser = argparse.ArgumentParser()
parser.add_argument("--frames_per_split", type=int, default=5, help="Number of frames on which the gradients were evaluated in every split.")
parser.add_argument('--num_splits', type=int, default=1, help='Number of splits (each containing frames_per_split frames.')
parser.add_argument('--systems', nargs='+',
                    help='List (separated by spaces) the names of the systems for which you wish to preprocess the data.',
                    required=True)
args = parser.parse_args()
data_pars = {system: DataParameters(dataset_name=system) for system in args.systems}

total_frames = args.frames_per_split*args.num_splits
FRAME_IDs = {system: random.sample(range(0, data_pars[system].num_frames), total_frames) for system in args.systems} # Pick frames on which to evaluate for each system

for job_no in trange(args.num_splits):
    with open(FRAMES_PER_JOBS_PATH_TEMPLATE.format(args.frames_per_split,job_no), 'w') as outfile:
        yaml.dump(FRAME_IDs[job_no*args.frames_per_split:(job_no+1)*args.frames_per_split], outfile, default_flow_style=False)