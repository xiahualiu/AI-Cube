import numpy as np
import torch
import torch.nn as nn

from Cube import cube222

class DQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQN, self).__init__()
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

    def forward(self, batch):
        x = batch.view(-1,self.input_size)
        body_output = self.body(batch)
        value_ouput = self.value(body_output)
        policy_output = self.policy(body_output)
        return policy_output, value_ouput

    def encode_cube(self,cube):
        unencoded=cube.state()
        encoded=np.zeros([8,3],dtype=np.uint8)
        
