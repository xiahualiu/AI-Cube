import numpy as np 

class cube222:
    # Keep track of top 4 stickers and bottom 4 stickers
        
    # Locations at beggining:
    # 1: [1,1,1] 
    # 2: [1,2,1]
    # 3: [2,2,1]
    # 4: [2,1,1]
    # 5: [1,1,2]
    # 6: [1,2,2]
    # 7: [2,2,1]
    # 8: [2,1,1]

    # Directions of the sticker:
    # 1: Top/Bottom 
    # 2: Left/Right 
    # 3: Front/Back

    def __init__(self, scram_num):
        # 8 blocks [location] [direction] - one hot encoding
        self.stickers=np.zeros([8,2],dtype=np.uint8)
        self.order=("111","121","221","211","112","122","222","212")
        self.Actions=("U","L","F","R","B","D")
        self._action_map_postion=(
            {0:1,1:2,2:3,3:0},
            {0:3,3:7,7:4,4:0},
            {3:2,2:6,6:7,7:3},
            {2:1,1:5,5:6,6:2},
            {1:0,0:4,4:5,5:1},
            {7:6,6:5,5:4,4:7})
        self._action_map_direction=(
            {1:1,2:3,3:2},
            {1:3,2:2,3:1},
            {3:3,1:2,2:1},
            {2:2,1:3,3:1},
            {3:3,1:2,2:1},
            {1:1,2:3,3:2})

        # Initial cube
        # Initial positions, also solved postions
        self.stickers[:,0]=np.array(range(8),dtype=np.uint8)
        # Initial directions, also solved directions
        self.stickers[:,1]=np.full([8],1,dtype=np.uint8)
        self.solved=np.copy(self.stickers)

        for i in range(scram_num):
            self.turn(np.random.permutation(6)[0]) 
            # Turn a random face

    def turn(self, action):
        for i in range(8):
            if self.stickers[i,0] in self._action_map_postion[action]:
                # Change the postion    
                self.stickers[i,0]=self._action_map_postion[action][self.stickers[i,0]]
                self.stickers[i,1]=self._action_map_direction[action][self.stickers[i,1]]
    
    def check(self):
        return np.array_equal(self.stickers, self.solved)

    def state(self):
        # This function's output will be feeded to the DQN network
        origin=np.copy(self.pattern)
        # Origin pattern
        