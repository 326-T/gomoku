#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
from collections import namedtuple
from collections import deque
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


class FNAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = None
        self.nn_controller = None
        self.estimate_probs = False
        self.initialized = False
        self.color = 1

    @classmethod
    def load(cls, model, epsilon=0.1):
        agent = cls(epsilon)
        agent.model = model
        agent.initialized = True
        return agent

    def estimate(self, state):
        x = self._append_color(state)
        return self.model.predict(x).detach().numpy()
        
    def policy(self, state, options):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.choice(options, 1)[0]
        else:
            estimates = self.estimate(state)[0]
            return self._choice_from_options(estimates, options)

    def replay(self, experiences, gamma=0.9, batch_size=128, epoch=2):
        states = np.vstack([e.s for e in experiences])
        next_states = np.vstack([e.n_s for e in experiences])

        estimateds = self.estimate(states)
        future = self.estimate(next_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        estimateds = np.array(estimateds)
        x = self._append_color(states)
        loss = self.model.train(x , estimateds, batch_size, epoch)
        return loss
    
    def _choice_from_options(self, estimates, options):
        index = np.argmax(estimates[options])
        return options[index]
    
    def _append_color(self, state):
        if state.ndim == 1:
            return np.block([state, self.color])
        else:
            return np.block([state, np.ones([state.shape[0], 1]) * self.color])


# In[ ]:




