"""
Model definitions for the active learning experiment.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import config

class BaseModel:
    """
    Base model class for active learning using MLPClassifier.
    """
    
    def __init__(self, hidden_layer_sizes=(100, 100), random_state=None,max_iter=300):

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class CommitteeModel:
    """
    Query-by-Committee model implementation.
    """
    
    def __init__(self, n_models, hidden_layer_sizes=(100, 100)):
        self.models = [
            BaseModel(hidden_layer_sizes=hidden_layer_sizes)
            for i in range(n_models)
        ]
        self.model= BaseModel(hidden_layer_sizes=hidden_layer_sizes,max_iter=300)
    
    def fit_committee(self, X, y):
        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):
        return np.array(self.model.predict(X))
    
    def fit(self,X,y):
        self.model.fit(X,y)
    
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
        n_classes = 10  # MNIST has 10 classes
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
