import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

from model import RL
from model import ExploreMemory

INPUT_SIZE=[7,24]
ACTIONS=6

device=torch.device("cpu")

# Build neural network
net = RL(INPUT_SIZE,ACTIONS).to(device)
print(net)
# Select optimizer
opt = optim.Adam(net.parameters(), lr=0.00005)
# Select loss function
memory=ExploreMemory()

loss_history=np.array([],dtype=np.float64)

for epoch in range(200):
    for i in [1,2,3,4,5,6,7,8]:
        memory.play(i)
        train_input, train_new_vals, train_new_acts=memory.process(net)
        opt.zero_grad()
        policy_out, value_out = net(train_input)
        policy_loss_t=F.cross_entropy(policy_out,train_new_acts)
        value_loss_t=(value_out-train_new_vals)**2
        value_loss_t=value_loss_t.mean()
        loss_t=value_loss_t+policy_loss_t
        loss_t.backward()
        print(loss_t)
        loss_history=np.append(loss_history, loss_t.detach().numpy())
        opt.step()
    print('Finish Epoch:{}!'.format(epoch))
    torch.save(net.state_dict(),'./trained_network.pt')
    sio.savemat('loss.mat',mdict={'loss': loss_history})
    
