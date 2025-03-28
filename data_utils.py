"""
Utilities for loading and preprocessing MNIST dataset with controlled label noise injection.
"""

import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import random

def load_mnist_data(train_length = 5000, test_length = 1000):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.reshape(-1))  # Flatten to 1D
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Convert to numpy arrays
    x_train_full = np.stack([data[0].numpy() for data in train_dataset])
    y_train_full = np.array([data[1] for data in train_dataset])
    x_test_full = np.stack([data[0].numpy() for data in test_dataset])
    y_test_full = np.array([data[1] for data in test_dataset])
    
    # Sample without replacement
    train_indices = np.random.choice(len(x_train_full), size=train_length, replace=False)
    test_indices = np.random.choice(len(x_test_full), size=test_length, replace=False)
    
    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_test = x_test_full[test_indices]
    y_test = y_test_full[test_indices]
    
    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test) # .transform is used here because we can't use any data from the test_set
    
    return x_train, y_train, x_test, y_test

def inject_label_noise(y, noise_percentage):
    '''Inject controlled noise into labels by randomly changing them'''

    num_classes = len(np.unique(y))
    y_noisy = y.copy()
    is_mislabeled = np.zeros(len(y), dtype=bool)
    
    # Calculate number of labels to noise
    num_to_noise = int(len(y) * noise_percentage / 100)
    
    # Randomly select indices to inject noise
    noise_idx = random.sample(range(len(y)), num_to_noise)
    
    for idx in noise_idx:
        # Get incorrect labels (excluding the true label)
        possible_labels = list(range(num_classes))
        possible_labels.remove(y[idx])
        
        # Randomly select an incorrect label
        y_noisy[idx] = np.random.choice(possible_labels)
        is_mislabeled[idx] = True
    
    return y_noisy, is_mislabeled

def create_initial_split(x, y, is_mislabeled, initial_size=100):
    # Randomly select initial indices
    total_samples = len(x)
    labeled_indices = np.random.choice(total_samples, initial_size, replace=False)
    unlabeled_indices = np.setdiff1d(range(total_samples), labeled_indices)
    
    # Split data
    x_labeled = x[labeled_indices]
    y_labeled = y[labeled_indices]
    x_unlabeled = x[unlabeled_indices]
    y_unlabeled = y[unlabeled_indices]
    is_mislabeled_labeled = is_mislabeled[labeled_indices]
    is_mislabeled_unlabeled = is_mislabeled[unlabeled_indices]
    
    return (x_labeled, y_labeled, x_unlabeled, y_unlabeled, 
            labeled_indices, unlabeled_indices, 
            is_mislabeled_labeled, is_mislabeled_unlabeled)