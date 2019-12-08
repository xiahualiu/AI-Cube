import numpy as np

def encode(state):

        # This is the function that encode one single state
        # We need to encode the [7,2] shaped state array into [1,168] shaped array
        # befor we feed it to the RL network
        #
        # Input Size: 1   Input Type: cube.state    (np.array.shape=[7,2])
        #     Output: 1) encoded_state    Size: [1,168]    Type:np.array    Dtype=np.float64
        #

    def f(x):
        result=np.zeros([24],dtype=np.float64)
        result[x[0]+x[1]*8]=1
        return result
    return np.array([f(xi) for xi in state],dtype=np.float64).flatten() # 1x168 array

def encode_batch(batch):

        # This is the function that encode a batch of states
        # We need to encode the [n,7,2] shaped state array into [n,1,168] shaped array
        # befor we feed it to the RL network
        #
        # Input Size: 1   Input Type: a batch of cube.state    (np.array.shape=[n,7,2])
        #     Output: 1) encoded_state_bathc    Size: [n,1,168]    Type:np.array    Dtype=np.float64

    def f(x):
        result=np.zeros([24],dtype=np.float64)
        result[x[0]+x[1]*8]=1
        return result
    return np.array([np.array([f(xi) for xi in state],dtype=np.float64).flatten() for state in batch],\
            dtype=np.float64)