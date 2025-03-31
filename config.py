"""
Configuration parameters for the active learning experiment with OCT dataset.
"""

# Dataset parameters
INITIAL_LABELED_SIZE = 40  # Number of initially labeled samples (10 per class)
NOISE_PERCENTAGES = [5]  # Test with 5% noise
BATCH_SIZE = 8  # Number of samples to query in each active learning iteration
NUM_ITERATIONS = 1  # Single iteration for testing
TRAIN_SIZE = 400  # 100 samples per class for initial testing
TEST_SIZE = 200  # 50 samples per class for initial testing

# Model parameters
COMMITTEE_SIZES = [3]  # Single committee size for testing

# Experiment parameters
EXPERIMENT_COUNTS = 1  # Single experiment for testing
