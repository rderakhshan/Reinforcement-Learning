"""
Deep Q-Network architecture implementation using TensorFlow/Keras.

This module provides the neural network architecture used by the TensorFlow DQN agent.
It implements the precise 3-layer convolutional network proposed in the
original DeepMind paper (Mnih et al., 2015) for processing Atari game frames natively.

Typical usage example:
    from network import DeepQNetwork
    net = DeepQNetwork(input_dims=(4, 84, 84), n_actions=6)
"""

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DeepQNetwork(keras.Model):
    """Convolutional Neural Network for Deep Q-Learning (TensorFlow Keras).

    Implements the core CNN architecture that evaluates pixel data and outputs 
    approximated Q-values utilizing native TensorFlow layers natively.

    Attributes:
        conv1 (tensorflow.keras.layers.Conv2D): First scaling layer natively (32 channels).
        conv2 (tensorflow.keras.layers.Conv2D): Second scaling layer natively (64 channels).
        conv3 (tensorflow.keras.layers.Conv2D): Third evaluating layer natively (64 channels).
        flat (tensorflow.keras.layers.Flatten): Routs multi-channel frames geometrically.
        fc1 (tensorflow.keras.layers.Dense): Fully connected Dense grouping logic logically.
        fc2 (tensorflow.keras.layers.Dense): Terminal prediction matrix returning absolute variables.
    """

    def __init__(self, input_dims, n_actions):
        """Initializes the TensorFlow Keras network sequentially parameters strictly.

        Args:
            input_dims (tuple): Absolute tensor constraints bounded natively.
            n_actions (int): Available logical actions mapping numerical sequences logically.
        """
        super(DeepQNetwork, self).__init__()
        
        # Geometrically scale frames utilizing channels_first explicitly matching PyTorch format
        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                            data_format='channels_first',
                            input_shape=input_dims)
        self.conv2 = Conv2D(64, 4, strides=(2, 2), activation='relu',
                            data_format='channels_first')
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu',
                            data_format='channels_first')
                            
        self.flat = Flatten()
        
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(n_actions, activation=None)

    def call(self, state):
        """Executes Keras feed-forward evaluation mathematically.

        Args:
            state (tf.Tensor): Batched inputs defining structural evaluation cleanly.

        Returns:
            tf.Tensor: Array outputs evaluating choices logically.
        """
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
