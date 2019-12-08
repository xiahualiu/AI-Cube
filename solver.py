import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from cube import cube
from model import RL


device=torch.device("cpu")

# This parameters should not be changed
INPUT_SIZE=[7,24]
ACTIONS=6

# Build solver network and feed state dict
net=RL(INPUT_SIZE, ACTIONS).to(device)
# Load the trained network
net.load_state_dict(torch.load('./trained_network.pt'))

# Initialize a cube
mycube=cube()

# Do some turns
# self.ACTIONS={0:"F", 1:"R", 2:"D", 3:"f", 4:"r", 5:"d"}
mycube.turn(1) # Turn R
mycube.turn(2) # Turn D
mycube.turn(3) # Turn f
mycube.turn(0) # Turn F

# Solve only by using neural network
while not mycube.check(mycube.state):
    # policy_predict, value_predict = net.predict_cube(mycube)
    # You can use state instead of cube
    policy_predict, value_predict = net.predict_state(mycube.state)
    max_val_t, max_act_t = policy_predict.max(dim=0)
    action=max_act_t.numpy()
    print(mycube.ACTIONS[int(action)])
    mycube.turn(action)    
pass