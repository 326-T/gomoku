#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,render_template,request
import numpy as np


# In[2]:


from modules.model import FCNN, FCNN_controller
from modules.env import Gomoku, Observer
from modules.agent import DNAgent, FNAgent


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
        if self.done or self.is_player_turn:
            return
        action = self.agent.policy(self.state, self.options)
        self.state, reward, self.done, self.options = self.env.step(self.player_ids["agent"], action)
        self.is_player_turn = True
        if reward == 1:
            self.result = "負け"
        elif reward == -0.3:
            self.result = "引き分け"
        
    def player_turn(self, action):
        if self.done or not self.is_player_turn:
            return
        self.state, reward, self.done, self.options = self.env.step(self.player_ids["player"], action)
        self.is_player_turn = False
        if reward == 1:
            self.result = "勝ち"
        elif reward == -0.3:
            self.result = "引き分け"


# In[4]:


def init_refree():
    env = Observer.load(Gomoku(3))
    fcnn = FCNN(env.dim_state+1, env.dim_action)
    model = FCNN_controller(fcnn)
    model.load_weight("data/dnn/1_model_fcnn")
    agent = DNAgent.load(model, None, 0)
    
    refree = Refree.load(env, agent)
    refree.reset()
    
    return refree


# In[5]:


refree = init_refree()


# In[6]:


app = Flask(__name__)

@app.route("/", methods=["GET"])
def root_get():
    refree.reset()
    return render_template("index.html", state=refree.env.state().reshape(3, 3).tolist(), options=refree.options.tolist(), 
    order=refree.player_ids["player"], result=refree.result)

@app.route("/", methods=["POST"])
def root_post():
    action = int(request.form.get("action"))
    refree.player_turn(action)
    refree.agent_turn()
    return render_template("index.html", state=refree.env.state().reshape(3, 3).tolist(), options=refree.options.tolist(), 
    order=refree.player_ids["player"], result=refree.result)
    


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




