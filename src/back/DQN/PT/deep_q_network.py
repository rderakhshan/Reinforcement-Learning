"""
Deep Q-Network architecture implementation using PyTorch.

This module provides the neural network architecture used by the DQN agent.
It implements the precise 3-layer convolutional network proposed in the
original DeepMind paper (Mnih et al., 2015) for processing Atari game frames.

Typical usage example:
    from deep_q_network import DeepQNetwork
    net = DeepQNetwork(lr=0.0001, n_actions=6, name='dqn',
                       input_dims=(4, 84, 84), chkpt_dir='models/')
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    """Convolutional Neural Network for Deep Q-Learning.

    Implements the core CNN architecture that evaluates pixel data and outputs 
    approximated Q-values. Provides utility for tracking hardware execution paths 
    and locally serializing generated weights to system disk.

    Attributes:
        checkpoint_dir (str): Directory where model weights are saved.
        checkpoint_file (str): Full path mapping to the model checkpoint file.
        conv1 (torch.nn.Conv2d): First convolutional layer (32 filters, 8x8 kernel).
        conv2 (torch.nn.Conv2d): Second convolutional layer (64 filters, 4x4 kernel).
        conv3 (torch.nn.Conv2d): Third convolutional layer (64 filters, 3x3 kernel).
        fc1 (torch.nn.Linear): Fully connected Dense mapping layer (512 units).
        fc2 (torch.nn.Linear): Final bounding Output layer grouping Q-values.
        optimizer (torch.optim.RMSprop): RMSprop optimizer for updating gradients.
        loss (torch.nn.MSELoss): Loss criterion mapped (Mean Squared Error).
        device (torch.device): Designated hardware execution component context.
    """

    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        """Initializes the DeepQNetwork model layers and parameter bounds.

        Args:
            lr (float): Extracted learning rate value utilized by the optimizer.
            n_actions (int): Boundaries mapping discrete operational actions limits.
            name (str): Identifier structuring formatting assigned saving files.
            input_dims (tuple): Shape of the input bounding tensor (C, H, W).
            chkpt_dir (str): Parent directory bounding persistent saving calls.
        """
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # First hidden convolutional layer structurally scales features widely
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        # Second hidden scaling evaluates narrower feature density bounds
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # Final grouped feature extraction layer preceding flat connections
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        """Calculates the flattened multi-dimensional parameter sizes computationally.

        Provides structural sizing parameters derived from compounding Conv2d layers
        without requiring physically hardcoded geometric parameters statically.

        Args:
            input_dims (tuple): The dimension parameters (C, H, W) of the input.

        Returns:
            int: The flattened feature magnitude dimensions mapped globally.
        """
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        """Executes feed-forward propagation cycles rendering states numerically.

        Args:
            state (torch.Tensor): A batched, normalized mapped input observation.

        Returns:
            torch.Tensor: Mapped action-value distributions validating valid choices.
        """
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        
        # Flattens tensors aligning spatial distributions linearly for fc tracking
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        """Commits locally held state-dictionaries to logical local physical drives."""
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Retains historically serialized models allocating previous weights actively."""
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
