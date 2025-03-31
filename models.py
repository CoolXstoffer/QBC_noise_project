"""
Model definitions for the active learning experiment using VGG16 for OCT classification.
"""

import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model, clone_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
import config

class BaseModel:
    """
    Base model class for active learning using VGG16 with transfer learning.
    """
    
    def __init__(self, random_state=None):
        # Set random seeds for reproducibility
        if random_state is not None:
            tf.random.set_seed(random_state)
            np.random.seed(random_state)

        # Load pre-trained VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze all base layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add new classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(4, activation='softmax')(x)  # 4 classes for OCT
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X, y):
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)

class CommitteeModel:
    """
    Query-by-Committee model implementation using VGG16.
    """
    
    def __init__(self, n_models):
        self.models = [BaseModel() for _ in range(n_models)]
        self.model = BaseModel()  # Main model for final predictions
    
    def fit_committee(self, X, y):
        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def vote_entropy(self, X):
        """
        Calculate committee disagreement for each sample.
        
        Args:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Disagreement scores
        """
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Calculate entropy of predictions for each sample
        n_classes = 4  # OCT has 4 classes
        disagreement_scores = []
        
        for sample_predictions in predictions.T:
            # Count votes for each class
            vote_counts = np.bincount(sample_predictions, minlength=n_classes)
            # Convert to probabilities
            probs = vote_counts / len(self.models)
            # Calculate entropy (with handling for zero probabilities)
            entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probs])
            disagreement_scores.append(entropy)
            
        return np.array(disagreement_scores)
