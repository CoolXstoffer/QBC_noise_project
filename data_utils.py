"""
Utilities for loading and preprocessing OCT dataset with controlled label noise injection.
"""

import numpy as np
import os
from PIL import Image
import random
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

def load_oct_data(data_path='/Users/cg/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Datasets/CellData/OCT', 
                  train_length=2000, test_length=500):
    """
    Load OCT dataset and preprocess for VGG16.
    
    Args:
        data_path (str): Path to OCT dataset
        train_length (int): Number of training samples to use
        test_length (int): Number of test samples to use
    """
    # Class mapping
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    
    def load_class_images(split_dir, class_name, max_samples=None):
        images = []
        class_path = os.path.join(data_path, split_dir, class_name)
        image_files = os.listdir(class_path)
        if max_samples:
            image_files = image_files[:max_samples]
        
        for img_file in image_files:
            if not img_file.startswith('.'):  # Skip hidden files
                img_path = os.path.join(class_path, img_file)
                # Load and preprocess image
                img = load_img(img_path, target_size=(224, 224))  # Resize to VGG16 input size
                img = img_to_array(img)
                if img.shape[-1] == 1:  # If grayscale
                    img = np.repeat(img, 3, axis=-1)  # Convert to RGB
                images.append(img)
        
        return np.array(images), np.full(len(images), label_encoder.transform([class_name])[0])
    
    # Load training data
    x_train_full = []
    y_train_full = []
    samples_per_class = train_length // len(classes)
    
    for class_name in classes:
        x_class, y_class = load_class_images('train', class_name, samples_per_class)
        x_train_full.append(x_class)
        y_train_full.append(y_class)
    
    x_train_full = np.concatenate(x_train_full)
    y_train_full = np.concatenate(y_train_full)
    
    # Load test data
    x_test_full = []
    y_test_full = []
    samples_per_class = test_length // len(classes)
    
    for class_name in classes:
        x_class, y_class = load_class_images('test', class_name, samples_per_class)
        x_test_full.append(x_class)
        y_test_full.append(y_class)
    
    x_test_full = np.concatenate(x_test_full)
    y_test_full = np.concatenate(y_test_full)
    
    # Random shuffle
    train_indices = np.random.permutation(len(x_train_full))
    test_indices = np.random.permutation(len(x_test_full))
    
    x_train = x_train_full[train_indices][:train_length]
    y_train = y_train_full[train_indices][:train_length]
    x_test = x_test_full[test_indices][:test_length]
    y_test = y_test_full[test_indices][:test_length]
    
    # Preprocess for VGG16
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    
    return x_train, y_train, x_test, y_test

def inject_label_noise(y, noise_percentage):
    '''Inject controlled noise into labels by randomly changing them'''
    num_classes = len(np.unique(y))
    y_noisy = y.copy()
    is_mislabeled = np.zeros(len(y), dtype=bool)
    
    if noise_percentage == 0:
        return y_noisy, is_mislabeled
    
    # Calculate number of labels to noise
    num_to_noise = max(1, int(len(y) * noise_percentage / 100))
    
    # Randomly select indices to inject noise
    noise_idx = random.sample(range(len(y)), num_to_noise)
    
    for idx in noise_idx:
        # Get incorrect labels (excluding the true label)
        possible_labels = list(range(num_classes))
        possible_labels.remove(y[idx])
        
        if possible_labels:  # Only inject noise if there are other classes to choose from
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
