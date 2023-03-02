"""
Specify the parameters regarding the data (e.g. number of residues of your protein) and the designed experiments
(how many Markov states to be estimated, how many models to estimate per dataset etc.)
"""

NUM_MODELS_PER_DATASET = 20
NUM_MARKOV_STATES = 3
NUM_INTERRESIDUE_DISTANCES = 780
NUM_RESIDUES = 42
SKIPPED_DIAG_WIDTH = 2

MAX_FRAMES = 650000
SPLIT_RATIO=0.9
MD_TIME_RESOLUTION_IN_NS=0.1
NETWORK_LAG_IN_FRAMES=50
TRAININGS_PER_SPLIT=3
NUM_TOTAL_TRAIN_ATTEMPTS=NUM_MODELS_PER_DATASET*TRAININGS_PER_SPLIT