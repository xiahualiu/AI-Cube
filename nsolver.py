import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from nMCTS import MCTS
from cube import cube
from model import RL

solution=[]

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

while not mycube.check(mycube.state):
    MCTS(mycube) #Utilize the MCTS to find next optimal move
pass
print(*solution, sep = "\n") #Once loop is completed, print solutions vector   

