"""
Utility for plotting experimental results.
"""

import json
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
import config

def load_latest_results():
    """Load results from multiple experiment runs."""
    results = []
    for experiment in range(1, config.EXPERIMENT_COUNTS + 1):
        file = f"results/experiment{experiment}_results.json"
        with open(file, 'r') as f:
            results.append(json.load(f))
    return results

def compute_statistics(results_list, key, metric):
    """
    Compute mean and standard deviation across multiple runs.
    
    Args:
        results_list (list): List of result dictionaries from different runs
        key (str): Key for specific experiment configuration
        metric (str): Metric to analyze
    
    Returns:
        tuple: (samples, mean_values, std_values)
    """
    # Get the number of samples (x-axis) from first run
    samples = results_list[0][key]['num_labeled_samples']
    
    # Stack values from all runs
    all_values = np.array([run[key][metric] for run in results_list])
    
    # Compute mean and std
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)
    
    return samples, mean_values, std_values

def plot_strategy_comparison(results_list, noise_percentage, metric='accuracy'):
    """
    Plot comparison of different strategies for a specific noise level.
    
    Args:
        results_list (list): List of experimental results from multiple runs
        noise_percentage (int): Noise level to compare
        metric (str): Which metric to plot ('accuracy' or 'mislabeled_ratio')
    """
    plt.figure(figsize=(10, 6))
    
    # Plot random sampling
    key = f"random_noise{noise_percentage}"
    samples, mean_values, std_values = compute_statistics(results_list, key, metric)
    line = plt.plot(samples, mean_values, label='Random Sampling', marker='o', markersize=4)
    color = line[0].get_color()
    
    # Plot QBC with different committee sizes
    committee_sizes = results_list[0]["config"]["COMMITTEE_SIZES"]
    for committee_size in committee_sizes:
        key = f"qbc{committee_size}_noise{noise_percentage}"
        samples, mean_values, std_values = compute_statistics(results_list, key, metric)
        line = plt.plot(samples, mean_values, 
                       label=f'QBC (size={committee_size})', 
                       marker='o', markersize=4)
        color = line[0].get_color()
    
    plt.xlabel('Number of Labeled Samples')
    ylabel = 'Accuracy' if metric == 'accuracy' else 'Ratio of Mislabeled Samples'
    plt.ylabel(ylabel)
    if metric == 'accuracy':
        plt.title(f'Strategy Comparison ({noise_percentage}% Noise) - {ylabel}\nShaded areas show ±1 std')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(f'plots/strategy_comparison_noise{noise_percentage}_{metric}.png')
    plt.close()

def main():
    """Generate all plots from multiple experiment runs."""
    results_list = load_latest_results()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot strategy comparisons for each noise level
    for noise in results_list[0]["config"]["NOISE_PERCENTAGES"]:
        plot_strategy_comparison(results_list, noise, metric='accuracy')
        plot_strategy_comparison(results_list, noise, metric='mislabeled_ratio')
    
    print("Plots have been generated in the plots directory.")

if __name__ == "__main__":
    main()
