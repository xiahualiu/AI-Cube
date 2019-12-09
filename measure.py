from MCTS import MCTS
from cube import cube
from model import RL
import torch
import numpy as np
import scipy.io as sio

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

def get_action(action):
    if action<3:
        counter=action+3
    else:
        counter=action-3
    return counter

record=np.zeros([50,2],dtype=np.uint32)

for depth in range(30):
    sio.savemat('record.mat',mdict={'record': record})
    for s in range(20):
        mycube=cube()
        for i in range(depth):
            action=np.random.randint(6)
            #print('Scrambling move: {}'.format(mycube.ACTIONS[action]))
            mycube.turn(action)
        # Here we start solving!!!
        # Reset fail_cnt

        fail_cnt=0
        solver=MCTS(mycube,net)

        moves=[]

        while not mycube.check(mycube.state):
            while solver.search_cnt < solver.max_search:
                solver.add_leaves()
                solver.search()

            if len(solver.best_path)==0:
                # We did not get anything
                fail_cnt+=1
                if fail_cnt>5:
                    record[depth][1]=record[depth][1]+1
                    print('Failed on {} steps!'.format(depth))
                    break
                if len(solver.best_node.lastactions)==0:
                    # If we did not EVEN find a best node
                    action=np.random.randint(6)
                else:
                    action=solver.best_node.lastactions[0]
                mycube.turn(action)
                del solver
                solver=MCTS(mycube,net)
                moves=moves+[action]
            else:
                # We find the way
                moves=moves+solver.best_path
                record[depth][0]=record[depth][0]+1
                print('Succeed on {} steps!'.format(depth))
                break;