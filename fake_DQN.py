# Fake neural network - for testing MCTS
import numpy as np

def fake_DQN(cube):
    # Must input 1 otherwise throw an error
    assert cube == 1
    # Return 6 random values, expected rewards
    return np.random.rand(6,1)