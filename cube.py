import numpy as np 

class cube(object):
    # Keep track of top 4 stickers and bottom 4 stickers

    def __init__(self):
        ## Some parameters 
        # Location List
        self.LOCATIONS={
            "111":0, "121":1, "221":2, "211":3, \
            "112":4, "122":5, "222":6, "212":7
            }
        # Direction List
        self.DIRECTIONS={"UP/DOWN":0, "LEFT/RIGHT":1, "FRONT/BACK":2}
        # Actions a cube can take
        self.ACTIONS={
            "U":0, "L":1, "F":2, "R":3, "B":4, "D":5, \
            "u":6, "l":7, "f":8, "r":9,"b":10,"d":11
            }
        # This is the ACTION when you print what you do
        # Change after rotate the whole cube
        self.R_ACTIONS={
            0:"U", 1:"L", 2:"F", 3:"R", 4:"B", 5:"D", \
            6:"u", 7:"l", 8:"f", 9:"r", 10:"b", 11:"d"
        }

        # How many stickers are we tracking
        self.STICKER_NUM=8

        # self.stickers have [location, direction] information of all stickers
        self.stickers=np.zeros([self.STICKER_NUM,2],dtype=np.uint8)
        self._action_map_postion=(
            {0:1,1:2,2:3,3:0}, # U
            {0:3,3:7,4:0,7:4}, # L
            {2:6,3:2,6:7,7:3}, # F
            {1:5,2:1,5:6,6:2}, # R
            {0:4,1:0,4:5,5:1}, # B
            {4:7,5:4,6:5,7:6}, # D
            {0:3,1:0,2:1,3:2}, # u
            {0:4,3:0,4:7,7:3}, # l
            {2:3,3:7,6:2,7:6}, # f
            {1:2,2:6,5:1,6:5}, # r
            {0:1,1:5,4:0,5:4}, # b
            {4:5,5:6,6:7,7:4}) # d
        self._action_map_direction=(
            {0:0,1:2,2:1}, # U
            {0:2,1:1,2:0}, # L
            {2:2,0:1,1:0}, # F
            {1:1,0:2,2:0}, # R
            {2:2,0:1,1:0}, # B
            {0:0,1:2,2:1}, # D
            {0:0,1:2,2:1}, # u
            {1:1,0:2,2:0}, # l
            {2:2,0:1,1:0}, # f
            {1:1,0:2,2:0}, # r
            {2:2,0:1,1:0}, # b
            {0:0,1:2,2:1}) # d

        # Initial positions, also solved postions
        self.stickers[:,0]=np.array(range(8),dtype=np.uint8)
        # Initial directions, also solved directions
        self.stickers[:,1]=np.full([8],0,dtype=np.uint8)
        self.solved=np.copy(self.stickers)

    # This can only be used by class itself
    def _turn(self, state, action):
        for i in range(8):
            # Change the postion
            if state[i,0] in self._action_map_postion[action]:
                state[i,0]=self._action_map_postion[action][state[i,0]]
                state[i,1]=self._action_map_direction[action][state[i,1]]
        return state
    
    def step(self, action):
        state=self._turn(self.state(), action)
        state=self.regularize(state)
        self.stickers=state

    def regularize(self,state):
        def s00(state):
            return state

        def s01(state):
            return s10(self.rotate_x_clockwise(state))

        def s02(state):
            return s30(self.rotate_y_anticlockwise(state))

        def s10(state):
            return self.rotate_z_anticlockwise(state) #1

        def s11(state):
            return self.rotate_x_anticlockwise(state) #2

        def s12(state):
            return s20(self.rotate_y_anticlockwise(state))

        def s20(state):
            return s30(self.rotate_z_clockwise(state))

        def s21(state):
            return s30(self.rotate_x_anticlockwise(state))

        def s22(state):
            return s10(self.rotate_y_clockwise(state))

        def s30(state):
            return self.rotate_z_clockwise(state) #3

        def s31(state):
            return s20(self.rotate_y_clockwise(state))

        def s32(state):
            return self.rotate_y_clockwise(state)#4

        def s40(state):
            return s01(self.rotate_y_anticlockwise(state))

        def s41(state):
            return self.rotate_x_clockwise(state) #5

        def s42(state):
            return self.rotate_y_anticlockwise(state) #6

        def s50(state):
            return s12(self.rotate_y_anticlockwise(state))
        
        def s51(state):
            return s10(self.rotate_x_anticlockwise(state))
        
        def s52(state):
            return s10(self.rotate_y_anticlockwise(state))

        def s60(state):
            return s22(self.rotate_y_clockwise(state))

        def s61(state):
            return s20(self.rotate_x_anticlockwise(state))

        def s62(state):
            return s20(self.rotate_y_clockwise(state))

        def s70(state):
            return s32(self.rotate_y_clockwise(state))

        def s71(state):
            return s30(self.rotate_x_clockwise(state))

        def s72(state):
            return s30(self.rotate_y_clockwise(state))

        lookup={
            (0,0):s00, (0,1):s01, (0,2):s02,
            (1,0):s10, (1,1):s11, (1,2):s12, 
            (2,0):s20, (2,1):s21, (2,2):s22,
            (3,0):s30, (3,1):s31, (3,2):s32,
            (4,0):s40, (4,1):s41, (4,2):s42,
            (5,0):s50, (5,1):s51, (5,2):s52,
            (6,0):s60, (6,1):s61, (6,0):s62,
            (7,0):s70, (7,1):s71, (7,2):s72,
        }
        func = lookup.get(tuple(state[0]),"Not Found!")
        state = func(state)
        assert np.array_equal(state[0],np.array([0,0],dtype=np.uint8))
        return state

    def check(self):
        return np.array_equal(self.stickers, self.solved)

    def neighbors(self):
        # Also output isgoal flag
        result=np.zeros((len(self.ACTIONS),8,2),dtype=np.uint8)
        isgoal=np.zeros((len(self.ACTIONS)), dtype=np.uint8)
        for action in range(len(self.ACTIONS)):
            temp=np.copy(self.stickers)
            temp=self._turn(temp, action)
            temp=self.regularize(temp)
            result[action]=temp
            isgoal[action] = np.array_equal(self.solved, temp)
        return (result, isgoal)

    def state(self):
        # This function's output will be feeded to the DQN network
        return self.stickers
        # Origin pattern
    
    # Whole cube rotate - for fix sticker #0
    def rotate_x_clockwise(self,state):
        state=self._turn(state,self.ACTIONS["F"])
        state=self._turn(state,self.ACTIONS["b"])
        old=self.R_ACTIONS.copy()
        self.R_ACTIONS={
            0:old[1], 1:old[5], 2:old[2], 3:old[0], 4:old[4], 5:old[3], \
            6:old[7], 7:old[11], 8:old[8], 9:old[6], 10:old[10], 11:old[9]
        }
        return state

    def rotate_x_anticlockwise(self,state):
        state=self._turn(state,self.ACTIONS["f"])
        state=self._turn(state,self.ACTIONS["B"])
        old=self.R_ACTIONS.copy()
        self.R_ACTIONS={
            0:old[3], 1:old[0], 2:old[2], 3:old[5], 4:old[4], 5:old[1], \
            6:old[9], 7:old[6], 8:old[8], 9:old[11], 10:old[10], 11:old[7]
        }
        return state

    def rotate_y_clockwise(self,state):
        state=self._turn(state,self.ACTIONS["R"])
        state=self._turn(state,self.ACTIONS["l"])
        old=self.R_ACTIONS.copy()
        self.R_ACTIONS={
            0:old[2], 1:old[1], 2:old[5], 3:old[3], 4:old[0], 5:old[4], \
            6:old[8], 7:old[7], 8:old[11], 9:old[9], 10:old[6], 11:old[10]
        }
        return state

    def rotate_y_anticlockwise(self,state):
        state=self._turn(state,self.ACTIONS["r"])
        state=self._turn(state,self.ACTIONS["L"])
        old=self.R_ACTIONS.copy()
        self.R_ACTIONS={
            0:old[4], 1:old[1], 2:old[0], 3:old[3], 4:old[5], 5:old[2], \
            6:old[10], 7:old[7], 8:old[6], 9:old[9], 10:old[11], 11:old[8]
        }
        return state

    def rotate_z_clockwise(self,state):
        state=self._turn(state,self.ACTIONS["U"])
        state=self._turn(state,self.ACTIONS["d"])
        old=self.R_ACTIONS.copy()
        self.R_ACTIONS={
            0:old[0], 1:old[2], 2:old[3], 3:old[4], 4:old[1], 5:old[5], \
            6:old[6], 7:old[8], 8:old[9], 9:old[10], 10:old[7], 11:old[11]
        }
        return state

    def rotate_z_anticlockwise(self,state):
        state=self._turn(state,self.ACTIONS["u"])
        state=self._turn(state,self.ACTIONS["D"])
        old=self.R_ACTIONS.copy()
        self.R_ACTIONS={
            0:old[0], 1:old[4], 2:old[1], 3:old[2], 4:old[3], 5:old[5], \
            6:old[6], 7:old[10], 8:old[7], 9:old[8], 10:old[9], 11:old[11]
        }
        return state

    # This function tells you what action you are going to take
    # corresponding to the original cube
    def print_move(self, action):
        return self.R_ACTIONS[action]
