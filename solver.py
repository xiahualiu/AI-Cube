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
DEPTH=4
for i in range(DEPTH)

def get_actions(action):
    if action<3
        conter=action+3
    else:
        conter=action-3
    action_set=np.array([0,1,2,3,4,5],dtype=np.uint32)
    return np.array([x for x in action_set if not x==conter], dtype=np.uint32)

old_move=99

# Solve only by using neural network
while not mycube.check(mycube.state):
    # policy_predict, value_predict = net.predict_cube(mycube)
    # You can use state instead of cube
    policy_predict, value_predict = net.predict_state(mycube.state)
    policy_predict=policy_predict[get_actions(old_move)]
    _, max_act_t = policy_predict.max(dim=0)
    action=max_act_t.numpy()
    old_move=action
    print(mycube.ACTIONS[int(action)])
    mycube.turn(action)    
pass