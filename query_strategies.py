"""
Implementation of different query strategies for active learning.
"""

import numpy as np
from models import CommitteeModel

class QueryStrategy:
    """Base class for query strategies."""
    
    def __init__(self, batch_size=10):
        """
        Initialize query strategy.
        
        Args:
            batch_size (int): Number of samples to query in each iteration
        """
        self.batch_size = batch_size
    
    def query(self, X_unlabeled, model):
        """
        Select instances to be labeled.
        
        Args:
            X_unlabeled (np.ndarray): Pool of unlabeled instances
            model: Trained model
            
        Returns:
            np.ndarray: Indices of selected instances
        """
        raise NotImplementedError("Subclasses must implement query method")

class RandomSampling(QueryStrategy):
    """Random sampling strategy."""
    
    def query(self, X_unlabeled, model=None):
        n_samples = len(X_unlabeled)
        return np.random.choice(
            n_samples,
            size=min(self.batch_size, n_samples),
            replace=False
        )

class QueryByCommittee(QueryStrategy):
    """Query-by-Committee strategy using entropy-based disagreement."""
    
    def __init__(self, n_models, batch_size=10):
        super().__init__(batch_size)
        self.n_models = n_models
    
    def query(self, X_unlabeled, model):
        if not isinstance(model, CommitteeModel):
            raise ValueError("QBC strategy requires a CommitteeModel")
        
        # Get disagreement scores for all unlabeled instances
        disagreement_scores = model.vote_entropy(X_unlabeled)
        
        # Select instances with highest disagreement
        print(disagreement_scores)
        top_indices = np.argsort(disagreement_scores)[-self.batch_size:]
        print(np.argsort(disagreement_scores))
        
        return top_indices
