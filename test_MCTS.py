from MCTS import MCTS
from cube import cube
from model import RL
import numpy as np
import torch


device=torch.device("cpu")

# This parameters should not be changed
INPUT_SIZE=[7,24]
ACTIONS=6

# Build solver network and feed state dict
net=RL(INPUT_SIZE, ACTIONS).to(device)
# Load the trained network
net.load_state_dict(torch.load('./trained_network.pt'))

mycube=cube()
# Do some turns
# self.ACTIONS={0:"F", 1:"R", 2:"D", 3:"f", 4:"r", 5:"d"}
DEPTH=3
for i in range(DEPTH):
    action=np.random.randint(6)
    print('Scrambling move: {}'.format(mycube.ACTIONS[action]))
    mycube.turn(action)

b = MCTS(mycube,net)
pass