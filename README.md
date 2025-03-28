<<<<<<< HEAD
# Active Learning with Noisy Labels Experiment

This project investigates the effectiveness of Query-by-Committee (QBC) for active learning in the presence of noisy labels. We use the MNIST dataset as a testbed and explore how different committee sizes and noise levels affect the learning performance.

## Experiment Overview

The experiment explores:
- Different noise levels: 5%, 10%, 15%, and 20% of labels are intentionally mislabeled
- Query strategies:
  - Query-by-Committee with varying committee sizes (2, 4, 8, 16 models)
  - Random Sampling (baseline)
- Performance metrics: Classification accuracy on clean test set

## Project Structure

```
├── data_utils.py      # Data loading and preprocessing utilities
├── models.py          # Model implementations (BaseModel and CommitteeModel)
├── query_strategies.py # Implementation of query strategies
├── config.py          # Configuration parameters
├── experiment.py      # Main experiment runner
├── plot_results.py    # Results visualization utilities
└── requirements.txt   # Project dependencies
```

## Requirements

To install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Experiments

1. Make sure all dependencies are installed
2. Run the main experiment:

```bash
python experiment.py
```

The results will be saved in the `results` directory with timestamps for each experiment run.

To visualize the results after running the experiment:

```bash
python plot_results.py
```

This will generate multiple plots in the `results` directory:
- Performance comparison across noise levels for each strategy
- Strategy comparison at each noise level
- Learning curves showing accuracy vs. number of labeled samples

## Implementation Details

### Data Processing
- MNIST dataset is loaded and preprocessed
- Features are standardized
- Labels are intentionally corrupted with specified noise percentages

### Models
- Base model: Multi-layer perceptron classifier
- Committee model: Ensemble of base models trained on bootstrapped data

### Query Strategies
1. **Query-by-Committee**
   - Multiple models vote on unlabeled instances
   - Disagreement is measured using vote entropy
   - Instances with highest disagreement are selected for labeling

2. **Random Sampling**
   - Baseline strategy
   - Randomly selects instances for labeling

### Active Learning Process
1. Start with small labeled pool
2. Train model(s) on labeled data
3. Query most informative samples from unlabeled pool
4. Update labeled and unlabeled pools
5. Repeat steps 2-4 for specified number of iterations

## Configuration

Key parameters can be modified in `config.py`:
- Initial labeled set size
- Noise percentages
- Committee sizes
- Number of active learning iterations
- Batch size for queries

## Results

Results are saved as JSON files containing:
- Accuracy trajectories
- Number of labeled samples
- Experiment metadata (noise level, strategy, committee size)

The results can be used to analyze:
- Learning curves under different noise levels
- Effectiveness of QBC vs random sampling
- Impact of committee size on query efficiency
- Robustness to label noise

## Notes

- Experiment uses scikit-learn's MLPClassifier for consistency and efficiency
- Random seed is fixed for reproducibility
- Each experiment configuration is run independently
=======
# QBC_noise_project
>>>>>>> d4194815601b4c736c3873fd02bd17c5bfbd4325
