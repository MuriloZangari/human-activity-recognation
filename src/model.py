"""
model.py

Defines the architecture for the Multilayer Perceptron (MLP) classifier
used to predict human activities from smartphone sensor data (UCI HAR v2).

The model is built using PyTorch and designed to accept:
- 561 input features
- 1 or more hidden layers with ReLU activation
- Dropout for regularization
- 6 output classes (multi-class classification)

Author: Murilo Zangari
"""

import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=561, hidden_dims=[256,128], output_dim=6, dropout_rate=0.3):
        """
        Initializes the MLPClassifier.

        Args:
            input_size (int): Number of input features.
            hidden_dims (list): List of integers representing the number of neurons in each hidden layer.
            output_dim (int): Number of output classes.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        current_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Create output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)