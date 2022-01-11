#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# In[2]:


class FCNN(nn.Module):
    
    def __init__(self, input_shape, output_shape, hidden_shape=[30, 30]):
        super(FCNN, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.model = nn.Sequential()
        if len(hidden_shape) == 0:
            self.model.add_module('fc1', nn.Linear(input_shape, output_shape))
            self.model.add_module('tanh1', nn.Tanh())
            
        else:
            self.model.add_module('fc1', nn.Linear(input_shape, hidden_shape[0]))
            self.model.add_module('tanh1', nn.Tanh())
            for i in range(1, len(hidden_shape)):
                self.model.add_module('fc'+str(i+1), nn.Linear(hidden_shape[i-1], hidden_shape[i]))
                self.model.add_module('tanh'+str(i+1), nn.Tanh())
            self.model.add_module('fc'+str(i+2), nn.Linear(hidden_shape[i], output_shape))
            self.model.add_module('tanh'+str(i+2), nn.Tanh())
        
    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = self.model(x)
        return x
        
    def loss(self, predicted, truth):
        return F.mse_loss(predicted, truth)


# In[3]:


class FCNN_controller():
    
    def __init__(self, model, device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train(self, x, y, batch_size, max_epoch):
        self.model.train()
        train_loss = 0
        for epoch in range(max_epoch):
            self.data_loader = torch.utils.data.DataLoader(GomokuDataset(x, y), batch_size = batch_size, shuffle=True)
            for x, y in (self.data_loader):
                x, y = x.to(torch.float).to(self.device), y.to(torch.float).to(self.device)
                self.optimizer.zero_grad()

                predicted = self.model(x)
                loss = self.model.loss(predicted, y)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
        
        return train_loss / (x.shape[0] * max_epoch)
        
    def predict(self, x):
        self.model.eval()
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        return self.model(x)        
        
    def save_weight(self, path='data/model_fcnn'):
        torch.save(self.model.state_dict(), path)
        
    def load_weight(self, path='data/model_fcnn'):
        self.model.load_state_dict(torch.load(path))


# In[4]:


class GomokuDataset(Dataset):
    
    def __init__(self, x, y, transform = None):
        self.x = x
        self.y = y
        self.transform = transform
        
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

