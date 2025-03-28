"""
Main experiment runner for active learning with noisy labels study.
"""

import numpy as np
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import os

from data_utils import load_mnist_data, inject_label_noise, create_initial_split
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
    start_time = time.time()
    # Initialize model
    if committee_size:
        model = CommitteeModel(
            n_models=committee_size, hidden_layer_sizes=config.HIDDEN_LAYER_SIZES
        )
    else:
        model = BaseModel(hidden_layer_sizes=config.HIDDEN_LAYER_SIZES)

    # Initialize results tracking
    results = {
        "accuracy": [],
        "num_labeled_samples": [],
        "num_mislabeled_selected": [],
        "mislabeled_ratio": [],
        "metadata": {
            "noise_percentage": noise_percentage,
            "strategy": query_strategy.__class__.__name__,
            "committee_size": committee_size,
            "timestamp": datetime.now().isoformat(),
            "initial_mislabeled_ratio": is_wrong_labeled.mean(),
        },
    }

    # Active learning loop
    for iteration in range(config.NUM_ITERATIONS):
        # Fit prediction model:
        model.fit(x_labeled,y_labeled)
        # Validate model:
        test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_pred)
        results["accuracy"].append(accuracy)
        results["num_labeled_samples"].append(len(x_labeled))
        results["num_mislabeled_selected"].append(int(is_wrong_labeled.sum()))
        results["mislabeled_ratio"].append(float(is_wrong_labeled.mean()))
        
        
        # If running with committee, fit committee models on data:
        if committee_size:
            model.fit_committee(x_labeled, y_labeled)

        # Query new samples
        query_indices = query_strategy.query(x_pool, model)

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
    print(
        f"Finished running experiment in {time.time() - start_time:.4f} seconds"
    )
    return results


def main():
    """Main function to run all experiments."""

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_mnist_data(
        train_length=config.TRAIN_SIZE, test_length=config.TEST_SIZE
    )

    # Create results directory
    os.makedirs("results", exist_ok=True)

    config_dict = {
        "INITIAL_LABELED_SIZE": config.INITIAL_LABELED_SIZE,
        "NOISE_PERCENTAGES": config.NOISE_PERCENTAGES,
        "BATCH_SIZE": config.BATCH_SIZE,
        "NOISE_PERCENTAGES": config.NOISE_PERCENTAGES,
        "NUM_ITERATIONS": config.NUM_ITERATIONS,
        "HIDDEN_LAYER_SIZES": config.HIDDEN_LAYER_SIZES,
        "COMMITTEE_SIZES": config.COMMITTEE_SIZES,
        "TRAIN_SIZE": config.TRAIN_SIZE,
        "TEST_SIZE": config.TEST_SIZE,
    }
    all_results = {"config": config_dict}

    # Experiment with different noise levels
    for experiment in range(1, config.EXPERIMENT_COUNTS + 1):
        start_experiment_time = time.time()
        print(
            f"Starting experiment {experiment} out of {config.EXPERIMENT_COUNTS} total"
        )

        for noise_percentage in config.NOISE_PERCENTAGES:
            print(f"\n Starting experiment with {noise_percentage}% of labels being mislabeled")
            # Add noise to y_train based on noise_percentage:
            y_train_noisy, is_mislabeled = inject_label_noise(y_train, noise_percentage=noise_percentage)

            # Create initial labeled dataset
            (x_labeled,y_labeled,x_pool,y_pool,labeled_indices,pool_indices,is_wrong_labeled,is_wrong_pool) = create_initial_split(X_train, y_train_noisy, is_mislabeled, config.INITIAL_LABELED_SIZE)

            random_strategy = RandomSampling(batch_size=config.BATCH_SIZE)

            results = run_experiment(
                x_labeled,
                y_labeled,
                x_pool,
                y_pool,
                X_test,
                y_test,
                is_wrong_labeled,
                is_wrong_pool,
                noise_percentage=noise_percentage,
                query_strategy=random_strategy,
            )
            all_results[f"random_noise{noise_percentage}"] = results

            # QBC experiments with different committee sizes
            for committee_size in config.COMMITTEE_SIZES:
                qbc_strategy = QueryByCommittee(
                    n_models=committee_size, batch_size=config.BATCH_SIZE
                )
                results = run_experiment(
                x_labeled,
                y_labeled,
                x_pool,
                y_pool,
                X_test,
                y_test,
                is_wrong_labeled,
                is_wrong_pool,
                noise_percentage=noise_percentage,
                query_strategy=qbc_strategy,
                committee_size=committee_size,
                )
                all_results[f"qbc{committee_size}_noise{noise_percentage}"] = results

        # Save results
        with open(f"results/experiment{experiment}_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n \n Experiment {experiment} finished in {time.time() - start_experiment_time:.4f} seconds \n \n")


if __name__ == "__main__":
    main()
