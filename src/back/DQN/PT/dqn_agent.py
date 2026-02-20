"""
Deep Q-Network Agent core logical interface binding states and parameters.

This module encapsulates the DQN agent methodologies connecting the Deep Q-Network,
Target Network, and underlying Replay Buffer memory models collectively.

Typical usage example:
    from dqn_agent import DQNAgent
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, ...)
    action = agent.choose_action(observation)
"""

import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent(object):
    """Deep Q-Learning Agent managing synchronized neural computations dynamically.

    Coordinates execution logic traversing the fundamental boundaries linking
    stochastic epsilon-greedy algorithms dynamically across decoupled Target mechanisms.

    Attributes:
        gamma (float): Discount rate parsing sequential reward parameters dynamically.
        epsilon (float): Dynamic random bounding limiting absolute algorithm convergence.
        lr (float): Agent's localized learning rate scaling structurally.
        n_actions (int): Discrete bounds referencing environment outputs mathematically.
        input_dims (tuple): Bounding dimensions framing structural inputs visually.
        batch_size (int): Total sampled parameter clusters routing memory gradients.
        eps_min (float): Lowest floor evaluated parsing epsilon random boundaries natively.
        eps_dec (float): Iterative step decrement subtracted from epsilon parameters natively.
        replace_target_cnt (int): Iterative ceiling tracking network synchronizations natively.
        action_space (list): Mapped valid integer representations available globally.
        learn_step_counter (int): Current counter evaluating iteration metrics dynamically.
        algo (str): Identifying string for model saved weights.
        env_name (str): Identifying string tracking the environment context.
        chkpt_dir (str): Base output directory mapping serialization logic natively.
        memory (ReplayBuffer): Retained caching structure containing historical logic.
        q_eval (DeepQNetwork): The actively predicting dynamically tracing network layers.
        q_next (DeepQNetwork): The structurally insulated target forecasting structure natively.
    """

    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        """Initializes agent configurations preparing external environments effectively.

        Args:
            gamma (float): Numeric discount referencing successive operational outcomes flexibly.
            epsilon (float): Evolving index triggering exploratory environment logic properly.
            lr (float): Assigned variable dictating scaling optimization properly.
            n_actions (int): Constant detailing output bounds explicitly properly.
            input_dims (tuple): Bounding array defining structural dimension limits.
            mem_size (int): Size constant scaling physical memory limits structurally.
            batch_size (int): Operational bounds targeting localized cluster quantities natively.
            eps_min (float, optional): Epsilon decay floor constants natively. Defaults to 0.01.
            eps_dec (float, optional): Rate of terminal decay variables natively. Defaults to 5e-7.
            replace (int, optional): Iteration synchronization maximum ceilings. Defaults to 1000.
            algo (str, optional): Classifying parameters routing logging algorithms. Defaults to None.
            env_name (str, optional): Identification tracking logging distributions. Defaults to None.
            chkpt_dir (str, optional): Directory assigning mapping targets statically. Defaults to 'tmp/dqn'.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        # Initialize the replay buffer
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # Initialize evaluation Q-Network
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        # Initialize target Q-Network
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        """Determines the appropriate action implementing an epsilon-greedy strategy.

        Args:
            observation (numpy.ndarray): The current environment frame state.

        Returns:
            int: Uniquely mapped decision mapping toward operational actions universally.
        """
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        """Stores the environment transition context into internal Replay Memory.

        Args:
            state (numpy.ndarray): Active tracking state representation before step.
            action (int): Discrete step taken triggering the transition.
            reward (float): Numeric progression variable resulting from the step.
            state_ (numpy.ndarray): Sequential outcome state representation after step.
            done (bool): Delineates strictly terminal boundaries ending iterations.
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """Draws randomized experience arrays distributing sequential tracking logically.

        Returns:
            tuple: Clustered PyTorch tensors formatted mapping memory parameters natively
                   (states, actions, rewards, states_, dones).
        """
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        """Synchronizes decoupling target networks resolving divergence logically.
        
        Evaluates current counter variables updating evaluating weights incrementally.
        """
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        """Decreases exploratory distribution bounds incrementally."""
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        """Locally caches structurally instantiated bounds universally."""
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        """Constructs and restores previously tracking environments properly."""
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        """Executes the foundational internal optimization looping gradients correctly.
        
        Resolves internal buffers sampling decoupled Q-Networks backpropagating globally.
        This handles the Huber loss calculation (by default MSE was specified, but
        functions identically for regression updates) utilizing PyTorch optimizers.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        # Update the target network if necessary
        self.replace_target_network()

        # Extract mini-batch dependencies from the ReplayBuffer Memory tracking arrays
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        # Calculates currently approximated metrics mapping action targets individually
        q_pred = self.q_eval.forward(states)[indices, actions]
        
        # Calculates sequentially projected target paths
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        # Halts forecasting sequential logic dynamically mapping limits universally
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        # Calculates internal regressions triggering optimization paths logically
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
