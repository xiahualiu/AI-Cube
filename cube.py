import numpy as np 
from encode import encode
from copy import deepcopy

class cube(object):
    # Keep track of top 4 stickers and bottom 4 stickers

    def __init__(self):
        ## Some parameters 
        # Location List 
        self.LOCATIONS={"121":0, "221":1, "211":2, "112":3, "122":4, "222":5, "212":6}
        # Direction List
        self.DIRECTIONS={"UP/DOWN":0, "LEFT/RIGHT":1, "FRONT/BACK":2}
        # Actions a cube can take
        self.ACTIONS={0:"F", 1:"R", 2:"D", 3:"f", 4:"r", 5:"d"}

        # How many stickers are we tracking
        # No.0 stikers removed (fixed at location 111 and direction up now)
        self.STICKER_NUM=7

        # self.stickers have [location, direction] information of all stickers
        self.state=np.zeros([self.STICKER_NUM,2],dtype=np.uint8)

        # Map - for turn function
        self._action_map_postion=(
            {1:5,2:1,5:6,6:2}, # F
            {0:4,1:0,4:5,5:1}, # R
            {3:6,4:3,5:4,6:5}, # D
            {1:2,2:6,5:1,6:5}, # f
            {0:1,1:5,4:0,5:4}, # r
            {3:4,4:5,5:6,6:3}) # d
        self._action_map_direction=(
            {2:2,0:1,1:0}, # F
            {1:1,0:2,2:0}, # R
            {0:0,1:2,2:1}, # D
            {2:2,0:1,1:0}, # f
            {1:1,0:2,2:0}, # r
            {0:0,1:2,2:1}) # d

        # Initial positions, also solved postions
        self.state[:,0]=np.array(range(7),dtype=np.uint8)
        # Initial directions, also solved directions
        self.state[:,1]=np.full([7],0,dtype=np.uint8)
        self.solved=np.copy(self.state)

    # This will change the cube itself
    def turn(self, action):
        for i in range(7):
            # Change the postion
            if self.state[i,0] in self._action_map_postion[action]:
                self.state[i,0]=self._action_map_postion[action][self.state[i,0]]
                self.state[i,1]=self._action_map_direction[action][self.state[i,1]]
    
    # Peek turn will not change the cube itself
    # Return the after state information
    def peek_turn(self, action):
        temp=np.copy(self.state)
        for i in range(7):
            # Change the postion
            if temp[i,0] in self._action_map_postion[action]:
                temp[i,0]=self._action_map_postion[action][temp[i,0]]
                temp[i,1]=self._action_map_direction[action][temp[i,1]]
        return temp

    # You must input the state (7x2, dtype=np.uint8)
    def check(self, state):
        return np.array_equal(state, self.solved)

    # Peek all will return an array of all next states
    # And if next state is solved
    # Also output isgoal flag

    def peek_all(self):
        next_states=np.zeros((len(self.ACTIONS),7,2),dtype=np.uint8)
        isgoal=np.zeros((len(self.ACTIONS)), dtype=np.uint8)
        for action in range(len(self.ACTIONS)):
            temp=self.peek_turn(action)
            next_states[action] = temp
            isgoal[action] = self.check(temp)
        return next_states, isgoal
    

    def print_move(self, action):
        # This function tells you what action you are going to take
        return self.ACTIONS[action]

    def new_cube(self, action):
    # This function will return a new cube based on the action and keep the old cube unchanged
        temp_cube=cube()
        temp_cube.state=np.copy(self.state)
        temp_cube.turn(action)
        return temp_cube

    def copy(self):
    # This function will return a copy of the current cube
        temp_cube=cube()
        temp_cube.state=np.copy(self.state)
        return temp_cube