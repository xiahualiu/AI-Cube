import numpy as np 

class Cube222:
    ## Cube has information on each side

    # Color is one-hot encoded (binary one-hot encoding)
    # White  : 1
    # Yellow : 10
    # Orange : 100
    # Red    : 1000
    # Green  : 10000
    # Blue   : 100000
        
    def __init__(self, scram_num):
        self.pattern=np.zeros([6,2,2])
        face_order=np.random.permutation(6)
        for i in range(6)
            self.pattern[i]=np.full([2,2],1<<face_order[i])
        for i in range(scram_num+1)
            self.turn(np.random.permutation(6)[0]) 
            # Turn a random face
        
    def turn(self, face):

