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


class Othello:
    
    def __init__(self, size=8):
        self.size = size
        self.dim_state = size ** 2
        self.dim_action = size ** 2
        self.reset()
        
    def reset(self):
        self.state = np.zeros([self.size, self.size])
        self.state[self.size//2-1][self.size//2-1], self.state[self.size//2][self.size//2] = 1, 1
        self.state[self.size//2][self.size//2-1], self.state[self.size//2-1][self.size//2] = -1, -1
        self.done = False
        self.reward = {1:0, -1:0}
        
    def step(self, player, x, y):
        reversible_lines = self._check_reversible_lines(player, x, y)
        if len(reversible_lines) != 0:
            self.state[y][x] = player
            for reversible_line in reversible_lines:
                for reversible_stone in reversible_line:
                    self.state[reversible_stone["y"]][reversible_stone["x"]] = player
        self._judge()
    
    def _check_reversible_lines(self, player, x, y):
        if self.state[y][x] != 0:
            return []
        reversible_lines = []
        for y_step in range(-1, 2):
            for x_step in range(-1, 2):
                stones = self._check_line(player, x, y, x_step, y_step)
                if len(stones) > 0:
                    reversible_lines.append(stones)
        return reversible_lines

 
    def _check_line(self, player, x, y, x_step, y_step):
        target = {"x": x + x_step, "y": y + y_step}
        reversible_line = []
        while 0 <= target["x"] and target["x"] < self.size and 0 <= target["y"] and target["y"] < self.size:
            if self.state[target["y"]][target["x"]] == 0:
                return []
            if self.state[target["y"]][target["x"]] == player:
                return reversible_line
            if self.state[target["y"]][target["x"]] == player * -1:
                reversible_line.append(target.copy())
            target["x"] += x_step
            target["y"] += y_step
        return []

    def _judge(self):
        if not (True in (self.state.reshape(-1)==1)):
            self._set_reward(-1)
            return
            
        if not (True in (self.state.reshape(-1)==-1)):
            self._set_reward(1)
            return

        if not (True in (self.state.reshape(-1)==0)):
            first = np.count_nonzero(self.state.reshape(-1)==1)
            second = np.count_nonzero(self.state.reshape(-1)==-1)
            if first > second:
                self._set_reward(1)
            elif first < second:
                self._set_reward(-1)
            else:
                self.reward[1] = -0.3
                self.reward[-1] = -0.3
                self.done = True
            return

    def _set_reward(self, winner):
        self.reward[winner] = 1
        self.reward[winner * -1] = -1
        self.done = True


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

# In[4]:


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

    def state(self):
        return self._env.state.reshape(-1)    
    
    def reward(self, player):
        return self._env.reward[player]
    
    def done(self):
        return self._env.done
    
    def reset(self):
        self._env.reset()
        return self.state(), self.options()

    def render(self):
        self._env.render()

    def step(self, player, action):
        self._env.step(player, action % self._env.size, action // self._env.size)
        return self.state(), self.reward(player), self.done(), self.options()   


# In[5]:


class OthelloObserver(Observer):

    def options(self, player):
        options = []
        for x in range(self._env.size):
            for y in range(self._env.size):
                if len(self._env._check_reversible_lines(player, x, y)) > 0:
                    options.append(self._env.size * y + x)
        return np.array(options)


    def reset(self):
        self._env.reset()
        return self.state(), self.options(1)

    def step(self, player, action):
        if action is not None:
            self._env.step(player, action % self._env.size, action // self._env.size)
        return self.state(), self.reward(player), self.done(), self.options(player*-1)   


