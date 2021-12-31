#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


class Gomoku:
    
    def __init__(self, size=3):
        self.size = size
        self.dim_state = size ** 2
        self.dim_action = size ** 2
        self.reset()
        
    def reset(self):
        self.state = np.zeros([self.size, self.size])
        self.done = False
        self.reward = {1:0, -1:0}
        
    def step(self, player, x, y):
        if not self._check_action(player, x, y):
            return
        self.state[y][x] = player
        self._judge()
    
    def _check_action(self, player, x, y):
        if self.done:
            return False
        if self.state[y][x] != 0:
            self.reward[player] = -1
            self.reward[player * -1] = 1
            self.done = True
            return False
        return True
    
    def _continue(self):
        return True in (self.state.reshape(-1)==0)
        
    def _judge(self):
        score = self._calc_score(1)
        opponent_score = self._calc_score(-1)
        if score == self.size:
            self.reward[1] = 1
            self.reward[-1] = -1
            self.done = True
            return
            
        if opponent_score == self.size:
            self.reward[1] = -1
            self.reward[-1] = 1
            self.done = True
            return
            
        if not self._continue():
            self.reward[1] = -0.3
            self.reward[-1] = -0.3
            self.done = True
            return
            
    def _calc_score(self, player):
        state = self.state * player
        row_col_score = np.max([np.sum(state, axis = 0), np.sum(state, axis = 1)])
        diagonal_score = max(state.trace(), state[:, ::-1].trace())
        return max(row_col_score, diagonal_score)
    
    def render(self):
        plt.figure(figsize=(6,6))
        plt.xlim([0, self.size])
        plt.ylim([0, self.size])
        plt.xticks(np.arange(self.size+1))
        plt.yticks(np.arange(self.size+1))
        plt.grid(which="major", color="black", alpha=0.5)
        white = np.where(self.state == 1)
        plt.scatter(white[1] + 0.5, white[0] + 0.5, color="white", s = 200, linewidths=2, ec="black")
        black = np.where(self.state == -1)
        plt.scatter(black[1] + 0.5, black[0] + 0.5, color="black", s = 200)
        plt.show()
        plt.close()


# In[3]:


class Observer():

    def __init__(self):
        self._env = None 
        self.dim_state = None
        self.dim_action = None

    @classmethod
    def load(cls, env):
        observer = cls()
        observer._env = env
        observer.dim_state = env.dim_state
        observer.dim_action = env.dim_action
        return observer
    
    def options(self):
        return np.where(self._env.state.reshape(-1) == 0)[0]


    def relative_state(self, player):
        return self._env.state.reshape(-1) * player
    
    def state(self):
        return self._env.state.reshape(-1)    
    
    def reward(self, player):
        return self._env.reward[player]
    
    def done(self):
        return self._env.done
    
    def reset(self):
        self._env.reset()
        return self.relative_state(1), self.options()

    def render(self):
        self._env.render()

    def step(self, player, action):
        self._env.step(player, action % self._env.size, action // self._env.size)
        return self.relative_state(player), self.reward(player), self.done(), self.options()   


# In[ ]:




