#!/usr/bin/env python
# coding: utf-8

# # Flappy Bird

# ## Imports

# In[23]:


import os
import time
import path
import random
import numpy as np
import glob
from collections import deque


# In[24]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[25]:


from ple.games.flappybird import FlappyBird
from ple import PLE


# ## Parameters

# In[26]:


n_episodes = 10000


# In[27]:


batch_size = 32


# In[28]:


base_path = 'flappy_weights_4'


# ## Agent

# In[29]:


class FBAgent():
    def __init__(self, input_size, n_actions, mem_size):
        self.input_size = input_size
        self.n_actions = n_actions
        
        self.learning_rate = 0.001
        self.gamma = 0.95
        
        self.epsilon = 0.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        
        self.memory = deque(maxlen=mem_size)
        
        self.model = self.__create_model()
        
    def __create_model(self):
        model = Sequential()
        
        model.add(Dense(30, input_shape=(self.input_size,), activation='tanh'))
        model.add(Dense(self.n_actions, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def predict_action(self, state):
        r = np.random.rand()
        if r <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            a = np.argmax(self.model.predict(state)[0])
            return a
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in minibatch:
            q_val = self.model.predict(state)
            q_next = self.model.predict(next_state)[0]
            target = q_val
            target[0][action] = (reward + self.gamma * np.max(q_next))
            
            self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, path):
        self.model.save_weights(path)
        
    def load(self, path):
        self.model.load_weights(path)


# ## Helping Functions

# In[30]:


def state_to_arr(state):
    keys = ['player_y', 'player_vel', 'next_pipe_dist_to_player', 'next_pipe_top_y', 'next_pipe_bottom_y']
#     keys = state.keys()
#     s = [state[x] for x in keys]
    s = []
    s.append(state['next_pipe_dist_to_player'])
    s.append(state['player_y'] - (state['next_pipe_top_y'] + state['next_pipe_bottom_y'])/2)
    l = len(s)
    return np.array(s).reshape((1, l))


# In[31]:


def use_agent(env, agent, state, score):
    a = agent.predict_action(state)
    action = actions[a]
    r = env.act(action)
    
    next_state = state_to_arr(env.getGameState())

    agent.remember(state, a, r, next_state)

    return next_state


# In[32]:


def get_last_weights(path):
    list_of_files = glob.glob(path + '/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# ## Environment Interaction

# In[33]:


game = FlappyBird(width=288, height=512, pipe_gap=100)
env = PLE(game, fps=30, frame_skip=1, num_steps=1, display_screen=True)
env.init()

actions = env.getActionSet()
state = state_to_arr(env.getGameState())
agent = FBAgent(len(state[0]), len(actions), mem_size=2000)

agent.load(get_last_weights(base_path))

if not os.path.exists(base_path):
    os.makedirs(base_path)

for i in range(n_episodes):
    env.reset_game()
    score = 0
    while not env.game_over():
        time.sleep(1/30)

        score += 1
        state = use_agent(env, agent, state, score)
        print('Episode: {}/{}, Score: {}, Epsilon: {:2f}'.format(i, n_episodes, score, agent.epsilon))
        
#         if len(agent.memory) > batch_size:
#             agent.train(batch_size)
        
#     agent.save(os.path.join('flappy_weights_5', 'weights'+ '_' + str(i)))

