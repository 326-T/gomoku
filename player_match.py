#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,render_template,request
import numpy as np


# In[2]:


from modules.model import FCNN, FCNN_controller
from modules.env import Gomoku, Observer, Othello, OthelloObserver
from modules.agent import FNAgent, DNAgent, DNQAgent


# In[3]:


class Refree:
    
    def __init__(self):
        self.env = None
        self.agent = None
        self.player_ids = None
        self.is_player_turn = False
        self.state = None
        self.options = None
        self.done = False
        self.result = "対戦中"
    
    @classmethod
    def load(cls, env, agent):
        refree = cls()
        refree.env = env
        refree.agent = agent
        return refree
    
    def reset(self):
        self.done = False
        self.result = "対戦中"
        self.state, self.options = self.env.reset()
        if np.random.random() < 0.5:
            self.player_ids = {"player" : 1, "agent" : -1}
            self.is_player_turn = True
        else:
            self.player_ids = {"player" : -1, "agent" : 1}
            self.is_player_turn = False
            self.agent_turn()
            
    def agent_turn(self):
        if self.done:
            return
        if len(self.options) == 0:
            self.state, reward, self.done, self.options = self.env.step(self.player_ids["agent"], None)
            return
        while True:
            action = self.agent.policy(self.state, self.options)
            self.state, reward, self.done, self.options = self.env.step(self.player_ids["agent"], action)
            if reward == -1:
                self.result = "勝ち"
                return
            elif reward == 1:
                self.result = "負け"
                return
            elif reward == -0.3:
                self.result = "引き分け"
                return
            if len(self.options) == 0:
                self.state, reward, self.done, self.options = self.env.step(self.player_ids["player"], None)
            else:
                self.is_player_turn = True
                return
            
    def player_turn(self, action):
        if self.done:
            return
        self.state, reward, self.done, self.options = self.env.step(self.player_ids["player"], action)
        self.is_player_turn = False
        if reward == 1:
            self.result = "勝ち"
        elif reward == -1:
            self.result = "負け"
        elif reward == -0.3:
            self.result = "引き分け"


# In[4]:


def init_gomoku_refree():
    env = Observer.load(Gomoku(3))
    fcnn = FCNN(env.dim_state+2, 1)
    model = FCNN_controller(fcnn)
    model.load_weight("data/gomoku/dnn/10_model_fcnn")
    agent = DNQAgent.load(model, None, 0)
    
    refree = Refree.load(env, agent)
    refree.reset()
    
    return refree


# In[5]:


def init_othello_refree():
    env = OthelloObserver.load(Othello())
    agent = DNQAgent(0)
    
    refree = Refree.load(env, agent)
    refree.reset()
    
    return refree


# In[6]:


gomoku_refree = init_gomoku_refree()
othello_refree = init_othello_refree()


# In[7]:


app = Flask(__name__)

@app.route("/", methods=["GET"])
def root_get():
    gomoku_refree.reset()
    return render_template("index.html", state=gomoku_refree.env.state().reshape(3, 3).tolist(), options=gomoku_refree.options.tolist(), 
    order=gomoku_refree.player_ids["player"], result=gomoku_refree.result)

@app.route("/", methods=["POST"])
def root_post():
    action = int(request.form.get("action"))
    gomoku_refree.player_turn(action)
    gomoku_refree.agent_turn()
    return render_template("index.html", state=gomoku_refree.env.state().reshape(3, 3).tolist(), options=gomoku_refree.options.tolist(), 
    order=gomoku_refree.player_ids["player"], result=gomoku_refree.result)
    

@app.route("/othello", methods=["GET"])
def othello_get():
    othello_refree.reset()
    return render_template("othello.html", state=othello_refree.env.state().reshape(8, 8).tolist(), options=othello_refree.options.tolist(), 
    order=othello_refree.player_ids["player"], result=othello_refree.result)

@app.route("/othello", methods=["POST"])
def othello_post():
    action = int(request.form.get("action"))
    othello_refree.player_turn(action)
    othello_refree.agent_turn()
    return render_template("othello.html", state=othello_refree.env.state().reshape(8, 8).tolist(), options=othello_refree.options.tolist(), 
    order=othello_refree.player_ids["player"], result=othello_refree.result)


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




