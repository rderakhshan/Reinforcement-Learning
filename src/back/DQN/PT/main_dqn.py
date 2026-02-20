"""
Main entry point orchestrating PyTorch DQN Atari training loops.

This script coordinates the execution pipeline connecting the Atari Gym
environment loops, action-predicting Agent networks, and data logging functions.

Typical usage example:
    python main_dqn.py
"""

import gymnasium as gym 
import numpy as np
import os
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gymnasium import wrappers

if __name__ == '__main__':
    # Initialize the base project boundaries and define saving locations dynamically
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Bootstraps the Atari environment with wrapper preprocessing functions
    env = make_env('PongNoFrameskip-v4')
    
    best_score = -np.inf
    load_checkpoint = False
    n_games = 250

    # Initializes Agent maintaining PyTorch Neural Network logic
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir=models_dir, algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    # Identifies generated graph parameters structurally
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = os.path.join(plots_dir, fname + '.png')

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation, info = env.reset()

        score = 0
        while not done:
            # Algorithmically determines trajectory behavior epsilon-greedily
            action = agent.choose_action(observation)
            
            # Submits chosen discrete outcome interacting with local game emulator
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            if not load_checkpoint:
                # Appends localized transition trajectories into replay buffers
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                # Calculates local gradients triggering optimization updates
                agent.learn()
                
            observation = observation_
            n_steps += 1
            
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        # Evaluates metric history generating persistence across weights periodically
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

        # Triggers visual HTML/Plotly web plotting tracking regression values sequentially
        x = [j+1 for j in range(len(scores))]
        plot_learning_curve(steps_array, scores, eps_history, figure_file)
