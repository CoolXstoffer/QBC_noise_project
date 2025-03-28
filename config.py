"""
Configuration parameters for the active learning experiment.
"""

# Dataset parameters
INITIAL_LABELED_SIZE = 30  # Number of initially labeled samples
NOISE_PERCENTAGES = [0,5,10,15,20] # Percentages of label noise to test
BATCH_SIZE = 10  # Number of samples to query in each active learning iteration
NUM_ITERATIONS = 20#50  # Number of active learning iterations
TRAIN_SIZE = 1000
TEST_SIZE = 250

# Model parameters
HIDDEN_LAYER_SIZES = (100, 100)  # Architecture for the base neural network
COMMITTEE_SIZES = [5, 10, 15, 20, 25]  # Different committee sizes to test

# Experiment parameters
RANDOM_SEED = 42  # For reproducibility
EXPERIMENT_COUNTS = 15 # Amount of experiments to run