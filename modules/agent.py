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
        x = self._append_turn(state)
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

        x = self._append_turn(states)
        loss = self.model.train(x , estimateds, batch_size, epoch)
        return loss
    
    def _choice_from_options(self, estimates, options):
        index = np.argmax(estimates[options])
        return options[index]
    
    def _append_turn(self, state):
        if state.ndim == 1:
            return np.block([state, self.color])
        else:
            return np.block([state, np.ones([state.shape[0], 1]) * self.color])

# In[3]:


class FNQAgent(FNAgent):

    def __init__(self, epsilon):
        super().__init__(epsilon)
        self.max_action = 1

    @classmethod
    def load(cls, model, max_action, epsilon=0.1):
        agent = super().load(cls, model, epsilon)
        agent.max_action = max_action
        return agent

    def estimate(self, state, action):
        x = self._append_turn_action(state, action)
        return self.model.predict(x).detach().numpy()
        
    def policy(self, state, options):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.choice(options, 1)[0]
        else:
            max_id = np.argmax([self.estimate(state, option)[0] for option in options])
            return options[max_id]

    def replay(self, experiences, gamma=0.9, batch_size=128, epoch=2):
        states = np.array([e.s for e in experiences])
        actions = np.array([e.a for e in experiences])

        estimateds = np.zeros([states.shape[0], 1])

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max([self.estimate(e.n_s, o) for o in e.n_o])
            estimateds[i][0] = reward

        x = self._append_turn_action(states, actions)
        loss = self.model.train(x, estimateds, batch_size, epoch)
        return loss
    
    def _append_turn_action(self, state, action):
        if state.ndim == 1:
            return np.block([state, self.color, action / self.max_action])
        else:
            return np.block([state, np.ones([state.shape[0], 1]) * self.color, action.reshape(-1, 1) / self.max_action])


# In[4]:


class DNAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = None
        self.target = None
        self.nn_controller = None
        self.estimate_probs = False
        self.initialized = False
        self.color = 1

    @classmethod
    def load(cls, model, target, max_action, epsilon=0.1):
        agent = cls(epsilon)
        agent.model = model
        agent.target = target
        agent.max_action = max_action
        agent.initialized = True
        return agent

    def estimate(self, state, use_target=False):
        x = self._append_turn(state)
        if use_target:
            return self.target.predict(x).detach().numpy()
        return self.model.predict(x).detach().numpy()
        
    def policy(self, state, options):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.choice(options, 1)[0]
        else:
            estimates = self.estimate(state)[0]
            return self._choice_from_options(estimates, options)

    def replay(self, experiences, gamma=0.9, batch_size=128, epoch=2):
        loss = 0
        dataset = NumpyDataset(experiences, batch_size)
        for es in dataset:
            states = np.array([e.s for e in es])
            next_states = np.array([e.n_s for e in es])
            actions = np.array([e.a for e in es])

            estimateds = self.estimate(states)
            future = self.estimate(next_states, use_target=True)

            for i, e in enumerate(es):
                reward = e.r
                if not e.d:
                    reward += gamma * np.max(future[i])
                estimateds[i][e.a] = reward

            x = self._append_turn(states)
            loss += self.model.train(x, estimateds, batch_size, epoch)

        self._hard_copy()
        return loss / len(dataset)
    
    def _choice_from_options(self, estimates, options):
        index = np.argmax(estimates[options])
        return options[index]
    
    def _append_turn(self, state):
        if state.ndim == 1:
            return np.block([state, self.color])
        else:
            return np.block([state, np.ones([state.shape[0], 1]) * self.color])

    def _hard_copy(self):
        self.target.model.load_state_dict(self.model.model.state_dict())

# In[5]:


class DNQAgent(DNAgent):

    def __init__(self, epsilon):
        super().__init__(epsilon)
        self.max_action = 1

    @classmethod
    def load(cls, model, target, max_action, epsilon=0.1):
        agent = super().load(model, target, epsilon)
        agent.max_action = max_action
        return agent

    def estimate(self, state, action, use_target=False):
        x = self._append_turn_action(state, action)
        if use_target:
            return self.target.predict(x).detach().numpy()
        return self.model.predict(x).detach().numpy()
        
    def policy(self, state, options):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.choice(options, 1)[0]
        else:
            max_id = np.argmax([self.estimate(state, option)[0] for option in options])
            return options[max_id]

    def replay(self, experiences, gamma=0.9, batch_size=128, epoch=2):
        loss = 0
        dataset = NumpyDataset(experiences, batch_size)
        for es in dataset:
            states = np.array([e.s for e in es])
            actions = np.array([e.a for e in es])

            estimateds = np.zeros([states.shape[0], 1])

            for i, e in enumerate(es):
                reward = e.r
                if not e.d:
                    reward += gamma * np.max([self.estimate(e.n_s, o, use_target=True) for o in e.n_o])
                estimateds[i][0] = reward

            x = self._append_turn_action(states, actions)
            loss += self.model.train(x, estimateds, batch_size, epoch)

        self._hard_copy()
        return loss / len(dataset)
    
    def _append_turn_action(self, state, action):
        if state.ndim == 1:
            return np.block([state, self.color, action / self.max_action])
        else:
            return np.block([state, np.ones([state.shape[0], 1]) * self.color, action.reshape(-1, 1) / self.max_action])

    def _hard_copy(self):
        self.target.model.load_state_dict(self.model.model.state_dict())

# In[6]:


class NumpyDataset():

    def __init__(self, x, batch_size):
        self.x = x.copy()
        if type(x) is deque:
            self.x = list(self.x)
            
        rng = np.random.default_rng()
        rng.shuffle(self.x, axis=0)
        self.batch_size = batch_size
        self.size = (len(x) - 1) // batch_size + 1

    def __getitem__(self, id):
        if id < 0 or self.__len__() <= id:
            raise IndexError
        max_id = min(self.batch_size * (id+1), len(self.x))
        return self.x[self.batch_size * id : max_id]

    def __len__(self):
        return self.size


if __name__ == "__main__":
    a = np.arange(12).reshape(3,4).tolist()
    for i in NumpyDataset(a, 2):
        print(i)