"""
Experience replay buffer modeling dynamically generated execution memories.

This module provides the ReplayBuffer class that handles caching logic for
experiences inherently standardizing algorithms across the TF application explicitly.

Typical usage example:
    from replay_memory import ReplayBuffer
    buffer = ReplayBuffer(1000000, (4, 84, 84), 6)
    buffer.store_transition(state, action, reward, next_state, done)
    states, actions, rewards, states_, dones = buffer.sample_buffer(32)
"""

import numpy as np


class ReplayBuffer(object):
    """Memory buffer managing discrete cyclic data mapping matrices dynamically.

    Utilizes numpy arrays mapping isolated arrays independently over standardized
    Python array-lists guaranteeing exceptionally performant tensor manipulation.

    Attributes:
        mem_size (int): Absolute maximum transition index mapped chronologically.
        mem_cntr (int): Index tracing chronological additions structuring loops.
        state_memory (numpy.ndarray): Pre-compiled buffer handling existing states.
        new_state_memory (numpy.ndarray): Pre-compiled buffer routing future states.
        action_memory (numpy.ndarray): Pre-compiled buffer logging decisions.
        reward_memory (numpy.ndarray): Pre-compiled buffer allocating rewards.
        terminal_memory (numpy.ndarray): Pre-compiled tracking limiting terminal checks.
    """

    def __init__(self, max_size, input_shape, n_actions):
        """Generates pre-allocated multidimensional components caching variables logically.

        Args:
            max_size (int): Volume threshold where chronological arrays begin resetting.
            input_shape (tuple): Observation shape constraints (e.g., 4 frames of 84x84).
            n_actions (int): Discrete bounds referencing environment outputs mathematically.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        
        # Maps dynamically evaluating numeric boundaries safely without memory scaling exceptions
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        """Inserts a structured operational sequence dynamically overriding stale loops.

        Identifies boundaries natively calculating modulo assignments wrapping indexes.

        Args:
            state (numpy.ndarray): Initial active representation constraints generated.
            action (int): Discrete index selected natively evaluating inputs realistically.
            reward (float): Floating numeric assessment evaluating active progress natively.
            state_ (numpy.ndarray): Subsequent frame sequence calculated sequentially.
            done (bool): Delineates strictly terminal boundaries completing cycle loops.
        """
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """Randomly targets randomized mini-batches tracking array indices numerically.

        Args:
            batch_size (int): Maximum quantity compiled parsing randomized dependencies.

        Returns:
            tuple: Clustered extraction representing sequential observations tracking logic
                   in the format (states, actions, rewards, new_states, dones).
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # Generates uniformly distributed indices implicitly avoiding identical duplicate values
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
