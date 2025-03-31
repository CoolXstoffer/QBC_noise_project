"""
Configuration parameters for the active learning experiment.
"""

# Dataset parameters
INITIAL_LABELED_SIZE = 50  # Number of initially labeled samples
NOISE_PERCENTAGES = [0,5,10,15,20] # Percentages of label noise to test
BATCH_SIZE = 10  # Number of samples to query in each active learning iteration
NUM_ITERATIONS = 50  # Number of active learning iterations
TRAIN_SIZE = 1500 # Size of training data subset
TEST_SIZE = 500 # SIze of testing data subset

# Model parameters
HIDDEN_LAYER_SIZES = (75, 75)  # Architecture for the base neural network
COMMITTEE_SIZES = [3, 6, 9, 12]  # Different committee sizes to test

# Experiment parameters
EXPERIMENT_COUNTS = 15 # Amount of experiments to run