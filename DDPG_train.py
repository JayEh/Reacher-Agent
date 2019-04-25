# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:50:29 2019

@author: jarre
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from ddpg_agent import Agent
from collections import deque
from unityagents import UnityEnvironment

def new_unity_environment():
    env = UnityEnvironment(file_name=".\\Reacher_Windows_x86_64\\Reacher.exe")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    
    # look up the size of the action and state spaces
    state_size = env_info.vector_observations[0].shape[0]
    action_size = brain.vector_action_space_size
    
    return (brain_name, env, env_info, state, state_size, action_size)

def ddpg_train(agent, env, brain_name, state_size, n_episodes=2500, max_t=1000):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    
    for i_episode in range(1, n_episodes+1):
        # reset the environment for the start of a new episode
        env_info = env.reset(train_mode=True)[brain_name] 
        # get the current state
        state = env_info.vector_observations[0]
        # the score each episode starts at zero
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                break 
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              
        mean_score = np.mean(scores_window)
                
        print('\rEpisode {}\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score))
        if np.mean(scores_window)>=30.0:  # solved is 30
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tActual Score:{:.2f}'.format(i_episode, mean_score, score))
            torch.save(agent.actor_local.state_dict(), 'Z:/{:.2f}_actor_checkpoint.pth'.format(mean_score))
            torch.save(agent.critic_local.state_dict(), 'Z:/{:.2f}_critic_checkpoint.pth'.format(mean_score))
            break  # or not and just keep on keepin on
    return scores

