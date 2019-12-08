import numpy as np

def encode(state):
    # input 7*2 np.array.uint8 output: 1*168 np.array.float64
    # use this before feed state to the network
    def f(x):
        result=np.zeros([24],dtype=np.float64)
        result[x[0]+x[1]*8]=1
        return result
    return np.array([f(xi) for xi in state],dtype=np.float64).flatten() # 1x168 array

def encode_batch(batch):
    # input n*7*2 batch of states
    def f(x):
        result=np.zeros([24],dtype=np.float64)
        result[x[0]+x[1]*8]=1
        return result
    return np.array([np.array([f(xi) for xi in state],dtype=np.float64).flatten() for state in batch],\
            dtype=np.float64)