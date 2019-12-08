import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube import cube
from encode import encode, encode_batch

class RL(nn.Module):
    def __init__(self, input_shape, action_size):
        super(RL, self).__init__()
        self.input_size = int(np.prod(input_shape))
        self.body = nn.Sequential(
            nn.Linear(self.input_size, 4096).double(),
            nn.ELU(),
            nn.Linear(4096,2048).double(),
            nn.ELU()
        )
        self.policy = nn.Sequential(
            nn.Linear(2048, 512).double(),
            nn.ELU(),
            nn.Linear(512,action_size).double()
        )
        self.value = nn.Sequential(
            nn.Linear(2048, 512).double(),
            nn.ELU(),
            nn.Linear(512,1).double()
        )
        self.memory = torch.tensor

    def forward(self, batch, value_only=False):
        body_output = self.body(batch) # input shape (n x 168, tensor)
        if value_only == True:
            return self.value(body_output)
        return self.policy(body_output), self.value(body_output)
        
    def predict_cube(self, cube):
        # Given a cube and the network will produce the value of the state and the policy 
        feed_item=torch.from_numpy(encode(cube.state))
        # encode rule - one hot rule
        return self.forward(feed_item)

    def predict_state(self, state):
        # Given a state and the network will produce the value of the state and the policy
        feed_item=torch.from_numpy(encode(temp))
        # encode rule - one hot rule
        return self.forward(feed_item)

BATCH_SIZE = 20
ACTIONS = 6
STICKERS = 7

class ExploreMemory(object):

    def __init__(self):
        self.now_memory = np.zeros((BATCH_SIZE,168),dtype=np.float64) 
        self.next_memory = np.zeros((BATCH_SIZE,ACTIONS,168),dtype=np.float64)
        self.reward_memory = np.zeros((BATCH_SIZE,ACTIONS), dtype=np.float64)
        self.cnt = 0

    def process(self,net):
        # This function will process the moves we generated during exploring
        # Produce new state values based on next states and bellman equation
        # Reset memory pointer after process

        self.cnt = 0
        # Read next_memory and feed to the network 
        feed_dict = torch.from_numpy(self.next_memory)
        feed_dict = feed_dict.view(-1, 168) 
        values_t = net(feed_dict,value_only=True)
        values_t = values_t.view(BATCH_SIZE, ACTIONS)
        # Zero goal
        values_t[np.nonzero(self.reward_memory)]=1
        values_t = values_t-1
        max_val_t, max_act_t = values_t.max(dim=1)
        # Make train inputs, new values, new actions
        train_input = torch.from_numpy(self.now_memory).detach()
        train_new_values = max_val_t.detach()
        train_new_actions = max_act_t.detach()
        return train_input.view(BATCH_SIZE, 168), train_new_values, train_new_actions

    def play(self, max_steps):
        # Play a series of moves on the initial cube and write into memory 
        while self.cnt<BATCH_SIZE:
            temp_cube=cube()
            for i in range(max_steps):
                # One random step on the cube
                temp_cube.turn(np.random.randint(ACTIONS))
                # Remeber the cube and continue
                temp_neighbors, temp_rewards = temp_cube.peek_all() 
                if self.cnt<BATCH_SIZE:
                    self.remember(encode(temp_cube.state), encode_batch(temp_neighbors), temp_rewards) 
    
    def remember(self, encoded_state, encoded_next_states, reward):
        self.now_memory[self.cnt]=encoded_state
        self.next_memory[self.cnt]=encoded_next_states
        self.reward_memory[self.cnt]=reward
        self.cnt += 1