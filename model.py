import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube import cube

class RL(nn.Module):
    def __init__(self, input_shape, action_size):
        super(RL, self).__init__()
        self.input_size = int(np.prod(input_shape))
        self.body = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ELU(),
            nn.Linear(4096,2048),
            nn.ELU()
        )
        self.policy = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512,action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512,1)
        )
        self.memory = torch.tensor

    def forward(self, batch, value_only=False):
        x = batch.view(-1,self.input_size)
        body_output = self.body(batch)
        if value_only == True:
            return self.value(body_output)
        return self.policy(body_output), self.value(body_output)

BATCH_SIZE = 128
ACTIONS = 12
STICKERS = 8

class ExploreMemory(object):

    def __init__(self):
        self.now_memory = np.zeros((BATCH_SIZE,7,2),dtype=np.uint8) 
        self.next_memory = np.zeros((BATCH_SIZE,ACTIONS,7,2),dtype=np.uint8)
        self.isgoal_memory = np.zeros((BATCH_SIZE,ACTIONS), dtype=np.uint8)
        self.cnt = 0

    def process(self,net):
        # This function will process the moves we generated during exploring
        # Produce new state values based on next states and bellman equation
        # Reset memory pointer after process

        self.cnt = 0
        # Read next_memory and feed to the network 
        feed_dict = torch.tensor(self.encode(self.next_memory.view().reshape(BATCH_SIZE*ACTIONS,7,2)),dtype=torch.float32)
        feed_dict = feed_dict.view(BATCH_SIZE*ACTIONS, 7*24) 
        values_t = net(feed_dict,value_only=True)
        values_t = values_t.view(BATCH_SIZE, ACTIONS)
        # Zero goal
        values_t = torch.tensor(self.isgoal_memory,dtype=torch.float32)+values_t-1
        max_val_t, max_act_t = values_t.max(dim=1)
        # Make train inputs, new values, new actions
        train_input = torch.tensor(self.encode(self.now_memory), dtype=torch.float32).detach()
        train_new_values = max_val_t.detach()
        train_new_actions = max_act_t.detach()
        return train_input, train_new_values, train_new_actions

    def play(self, max_steps):
        # Play a series of moves on the initial cube and write into memory 
        while self.cnt<128:
            temp_cube=cube()
            for i in range(max_steps):
                # One random step on the cube
                temp_cube.step(np.random.randint(ACTIONS))
                # Remeber the cube and continue
                if self.cnt<128:
                    self.remember(temp_cube.state(), temp_cube.neighbors()[0], temp_cube.neighbors()[1]) 
    
    def remember(self, state, next_states, isgoal):
        self.now_memory[self.cnt]=state[1:8,:]
        self.next_memory[self.cnt]=next_states[:,1:8,:]
        self.isgoal_memory[self.cnt]=isgoal
        self.cnt += 1
    
    def encode(self, batch):
        # encode rule - one hot rule
        def f(x):
            result=np.zeros([24],dtype=np.float32)
            result[x[0]+x[1]*8]=1
            return result
        return np.array([np.array([f(xi) for xi in state],dtype=np.float32) for state in batch],\
            dtype=np.float32)
