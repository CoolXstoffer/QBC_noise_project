"""
Main experiment runner for active learning with noisy labels study using OCT dataset.
"""

import numpy as np
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import os
import tensorflow as tf
import copy

from data_utils import load_oct_data, inject_label_noise, create_initial_split
from models import CommitteeModel, BaseModel
from query_strategies import RandomSampling, QueryByCommittee
import config
import time

def run_experiment(
    x_labeled,
    y_labeled,
    x_pool,
    y_pool,
    X_test,
    y_test,
    is_wrong_labeled,
    is_wrong_pool,
    noise_percentage,
    query_strategy,
    committee_size=None,
):
    print(f"Running experiment with strategy={query_strategy.__class__.__name__} with {query_strategy.n_models if hasattr(query_strategy,'n_models') else ''} committee members")
    experiment_start_time = time.time()

    # Initialize model
    if committee_size:
        model = CommitteeModel(n_models=committee_size)
    else:
        model = BaseModel()

    # Initialize results tracking
    results = {
        "accuracy": [],
        "num_labeled_samples": [],
        "num_mislabeled_selected": [],
        "mislabeled_ratio": [],
        "iteration_times": [],
        "metadata": {
            "noise_percentage": noise_percentage,
            "strategy": query_strategy.__class__.__name__,
            "committee_size": committee_size,
            "timestamp": datetime.now().isoformat(),
            "initial_mislabeled_ratio": is_wrong_labeled.mean(),
            "total_experiment_time": None,
        },
    }

    # Active learning loop
    for iteration in range(config.NUM_ITERATIONS):
        iteration_start_time = time.time()
        print(f"\nIteration {iteration+1}/{config.NUM_ITERATIONS}")
        
        # Fit prediction model
        print("Training model...")
        model_fit_start = time.time()
        model.fit(x_labeled, y_labeled)
        model_fit_time = time.time() - model_fit_start
        print(f"Model training took {model_fit_time:.2f} seconds")

        # Validate model
        print("Evaluating model...")
        eval_start = time.time()
        test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_pred)
        eval_time = time.time() - eval_start
        print(f"Evaluation took {eval_time:.2f} seconds")

        results["accuracy"].append(accuracy)
        results["num_labeled_samples"].append(len(x_labeled))
        results["num_mislabeled_selected"].append(int(is_wrong_labeled.sum()))
        results["mislabeled_ratio"].append(float(is_wrong_labeled.mean()))
        
        print(f"Current accuracy: {accuracy:.4f}")
        
        # If running with committee, fit committee models on data
        if committee_size:
            print("Training committee models...")
            committee_start = time.time()
            model.fit_committee(x_labeled, y_labeled)
            committee_time = time.time() - committee_start
            print(f"Committee training took {committee_time:.2f} seconds")

        # Query new samples
        print("Selecting new samples...")
        query_start = time.time()
        query_indices = query_strategy.query(x_pool, model)
        query_time = time.time() - query_start
        print(f"Sample selection took {query_time:.2f} seconds")

        if len(query_indices) == 0:
            break

        # Update labeled and unlabeled pools
        new_labeled_x = x_pool[query_indices]
        new_labeled_y = y_pool[query_indices]
        new_mislabeled = is_wrong_pool[query_indices]

        x_labeled = np.vstack((x_labeled, new_labeled_x))
        y_labeled = np.concatenate((y_labeled, new_labeled_y))
        is_wrong_labeled = np.concatenate((is_wrong_labeled, new_mislabeled))

        x_pool = np.delete(x_pool, query_indices, axis=0)
        y_pool = np.delete(y_pool, query_indices)
        is_wrong_pool = np.delete(is_wrong_pool, query_indices)

        # Record iteration time
        iteration_time = time.time() - iteration_start_time
        results["iteration_times"].append(iteration_time)
        print(f"Iteration {iteration+1} completed in {iteration_time:.2f} seconds")

    total_time = time.time() - experiment_start_time
    results["metadata"]["total_experiment_time"] = total_time
    print(f"Finished running experiment in {total_time:.2f} seconds")
    return results

def run_full_experiment(experiment_num, X_train, y_train, X_test, y_test):
    """Run a complete experiment with all noise levels and strategies."""
    experiment_results = {}
    
    for noise_percentage in config.NOISE_PERCENTAGES:
        print(f"\nRunning with noise level: {noise_percentage}%")
        
        # Add noise to y_train based on noise_percentage
        y_train_noisy, is_mislabeled = inject_label_noise(y_train, noise_percentage=noise_percentage)

        # Create initial labeled dataset
        (x_labeled, y_labeled, x_pool, y_pool, labeled_indices, pool_indices,
         is_wrong_labeled, is_wrong_pool) = create_initial_split(
             X_train, y_train_noisy, is_mislabeled, config.INITIAL_LABELED_SIZE
        )

        # Store initial dataset state for different strategies
        initial_state = {
            'x_labeled': copy.deepcopy(x_labeled),
            'y_labeled': copy.deepcopy(y_labeled),
            'x_pool': copy.deepcopy(x_pool),
            'y_pool': copy.deepcopy(y_pool),
            'is_wrong_labeled': copy.deepcopy(is_wrong_labeled),
            'is_wrong_pool': copy.deepcopy(is_wrong_pool)
        }

        # Random sampling strategy
        random_strategy = RandomSampling(batch_size=config.BATCH_SIZE)
        results = run_experiment(
            initial_state['x_labeled'],
            initial_state['y_labeled'],
            initial_state['x_pool'],
            initial_state['y_pool'],
            X_test,
            y_test,
            initial_state['is_wrong_labeled'],
            initial_state['is_wrong_pool'],
            noise_percentage=noise_percentage,
            query_strategy=random_strategy,
        )
        experiment_results[f"random_noise{noise_percentage}"] = results

        # QBC experiments with different committee sizes
        for committee_size in config.COMMITTEE_SIZES:
            print(f"\nRunning QBC with committee size: {committee_size}")
            qbc_strategy = QueryByCommittee(
                n_models=committee_size, batch_size=config.BATCH_SIZE
            )
            results = run_experiment(
                copy.deepcopy(initial_state['x_labeled']),
                copy.deepcopy(initial_state['y_labeled']),
                copy.deepcopy(initial_state['x_pool']),
                copy.deepcopy(initial_state['y_pool']),
                X_test,
                y_test,
                copy.deepcopy(initial_state['is_wrong_labeled']),
                copy.deepcopy(initial_state['is_wrong_pool']),
                noise_percentage=noise_percentage,
                query_strategy=qbc_strategy,
                committee_size=committee_size,
            )
            experiment_results[f"qbc{committee_size}_noise{noise_percentage}"] = results
    
    return experiment_results

def main():
    """Main function to run all experiments."""
    overall_start_time = time.time()
    print("Loading OCT dataset...")
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_oct_data(
        train_length=config.TRAIN_SIZE, test_length=config.TEST_SIZE
    )

    # Create results directory
    os.makedirs("results", exist_ok=True)

    config_dict = {
        "INITIAL_LABELED_SIZE": config.INITIAL_LABELED_SIZE,
        "NOISE_PERCENTAGES": config.NOISE_PERCENTAGES,
        "BATCH_SIZE": config.BATCH_SIZE,
        "NUM_ITERATIONS": config.NUM_ITERATIONS,
        "COMMITTEE_SIZES": config.COMMITTEE_SIZES,
        "TRAIN_SIZE": config.TRAIN_SIZE,
        "TEST_SIZE": config.TEST_SIZE,
    }
    
    # Set memory growth for GPUs if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Run multiple experiments
    for experiment in range(1, config.EXPERIMENT_COUNTS + 1):
        experiment_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Starting experiment {experiment} out of {config.EXPERIMENT_COUNTS}")
        print(f"{'='*50}")

        # Run the experiment
        experiment_results = run_full_experiment(experiment, X_train, y_train, X_test, y_test)
        
        # Add configuration and timing information
        experiment_results["config"] = config_dict
        experiment_results["metadata"] = {
            "experiment_number": experiment,
            "timestamp": datetime.now().isoformat(),
            "experiment_duration": time.time() - experiment_start_time
        }

        # Save results
        results_path = f"results/experiment{experiment}_results.json"
        with open(results_path, "w") as f:
            json.dump(experiment_results, f, indent=2)
        print(f"\nExperiment {experiment} results saved to {results_path}")
        print(f"Experiment duration: {time.time() - experiment_start_time:.2f} seconds")

    overall_time = time.time() - overall_start_time
    print(f"\nAll experiments completed in {overall_time:.2f} seconds")

if __name__ == "__main__":
    main()
