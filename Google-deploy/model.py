import torch
import torch.nn as nn
import numpy as np


class FFNClassifier(nn.Module):
    def __init__(self, input_dim, architecture):
        super(FFNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in architecture:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.5),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, X):
        """
        Predict class probabilities for input X

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array-like of shape (n_samples, 2)
            Returns predicted probabilities for both classes [P(y=0), P(y=1)]
        """
        # convert X to numpy array
        X = X.to_numpy()

        # Convert input to tensor if not already
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        # Set model to evaluation mode
        self.eval()

        # Get predictions
        with torch.no_grad():
            probas = self.forward(X).numpy()

        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.column_stack([1 - probas, probas])
