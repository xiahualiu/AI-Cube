import numpy as np 

class cube222:
    ## Cube has information on each side

    # Color is one-hot encoded (binary one-hot encoding)
    # White   : 1
    # Orange  : 10
    # Green   : 100
    # Red     : 1000
    # Blue    : 10000
    # Yellow  : 100000
        
    def __init__(self, scram_num):
        self.pattern=np.zeros([6,2,2],dtype=np.uint32)
        self.color=('W','O','G','R','B','Y')
        self.face=("U","L","F","R","B","D")
        for i in range(6):
            self.pattern[i]=np.full([2,2],i,dtype=np.uint32)
        for i in range(scram_num):
            self.turn(np.random.permutation(6)[0]) 
            # Turn a random face
        
    def turn(self, face):
        if face==0:
            self.pattern[0]=np.array(list(zip(*self.pattern[0][::-1])))
            temp=np.copy(self.pattern[1][0])
            self.pattern[1][0]=self.pattern[2][0]
            self.pattern[2][0]=self.pattern[3][0]
            self.pattern[3][0]=self.pattern[4][0]
            self.pattern[4][0]=temp
        elif face==1:
            self.pattern[1]=np.array(list(zip(*self.pattern[1][::-1])))
            temp=np.copy(self.pattern[0][:,0])
            self.pattern[0][:,0]=self.pattern[4][:,1][::-1]
            self.pattern[4][:,1]=self.pattern[5][:,0][::-1]
            self.pattern[5][:,0]=self.pattern[2][:,0]
            self.pattern[2][:,0]=temp
        elif face==2:
            self.pattern[2]=np.array(list(zip(*self.pattern[2][::-1])))
            temp=np.copy(self.pattern[1][:,1])
            self.pattern[1][:,1]=self.pattern[5][0]
            self.pattern[5][0]=self.pattern[3][:,0][::-1]
            self.pattern[3][:,0]=self.pattern[0][1]
            self.pattern[0][1]=temp[::-1]
        elif face==3:
            self.pattern[3]=np.array(list(zip(*self.pattern[3][::-1])))
            temp=np.copy(self.pattern[2][:,1])
            self.pattern[2][:,1]=self.pattern[5][:,1]
            self.pattern[5][:,1]=self.pattern[4][:,0][::-1]
            self.pattern[4][:,0]=self.pattern[0][:,1][::-1]
            self.pattern[0][:,1]=temp
        elif face==4:
            self.pattern[4]=np.array(list(zip(*self.pattern[4][::-1])))
            temp=np.copy(self.pattern[3][:,1])
            self.pattern[3][:,1]=self.pattern[5][1][::-1]
            self.pattern[5][1]=self.pattern[1][:,0]
            self.pattern[1][:,0]=self.pattern[0][0][::-1]
            self.pattern[0][0]=temp
        elif face==5:
            self.pattern[5]=np.array(list(zip(*self.pattern[5][::-1])))
            temp=np.copy(self.pattern[1][1])
            self.pattern[1][1]=self.pattern[4][1]
            self.pattern[4][1]=self.pattern[3][1]
            self.pattern[3][1]=self.pattern[2][1]
            self.pattern[2][1]=temp
        else:
            print("Not valid face number")
            exit()

    def print(self):
        # First line
        print('  {}{}    '.format(self.color[self.pattern[0][0][0]],self.color[self.pattern[0][0][1]]))
        print('  {}{}    '.format(self.color[self.pattern[0][1][0]],self.color[self.pattern[0][1][1]]))
        print('{}{}{}{}{}{}{}{}'.format(self.color[self.pattern[1][0][0]], \
            self.color[self.pattern[1][0][1]], self.color[self.pattern[2][0][0]], \
            self.color[self.pattern[2][0][1]], self.color[self.pattern[3][0][0]], \
            self.color[self.pattern[3][0][1]], self.color[self.pattern[4][0][0]], \
            self.color[self.pattern[4][0][1]]))
        print('{}{}{}{}{}{}{}{}'.format(self.color[self.pattern[1][1][0]], \
            self.color[self.pattern[1][1][1]], self.color[self.pattern[2][1][0]], \
            self.color[self.pattern[2][1][1]], self.color[self.pattern[3][1][0]], \
            self.color[self.pattern[3][1][1]], self.color[self.pattern[4][1][0]], \
            self.color[self.pattern[4][1][1]]))
        print('  {}{}    '.format(self.color[self.pattern[5][0][0]],self.color[self.pattern[5][0][1]]))
        print('  {}{}    '.format(self.color[self.pattern[5][1][0]],self.color[self.pattern[5][1][1]]))

    def check_face(self,face):
        temp=self.pattern[face]
        return np.array_equal(np.full([2,2],temp[0][0]), temp)

    def check(self):
        result=True;
        for i in range(6):
            result=result and self.check_face(i) 
        return result
        # Only output true when all faces are checked true

    def rotate_h(self):
        # Rotating the whole cube 90 degree clockwise horizontally 
        temp_face=np.copy(self.pattern[1])
        self.pattern[0]=np.array(list(zip(*self.pattern[0][::-1])))
        self.pattern[5]=np.array(list(zip(*self.pattern[5])[::-1]))
        self.pattern[1]=self.pattern[2]
        self.pattern[2]=self.pattern[3]
        self.pattern[3]=self.pattern[4]
        self.pattern[4]=temp_face

    def rotate_v(self):
        # Rotating the whole cube 90 degree clockwise vertically
        temp_face=np.copy(self.pattern[1])
        self.pattern[2]=np.array(list(zip(*self.pattern[2][::-1])))
        self.pattern[4]=np.array(list(zip(*self.pattern[4])[::-1]))
        self.pattern[1]=np.array(list(zip(*self.pattern[5][::-1])))
        self.pattern[5]=np.array(list(zip(*self.pattern[3][::-1])))
        self.pattern[3]=np.array(list(zip(*self.pattern[0][::-1])))
        self.pattern[0]=np.array(list(zip(*temp_face[::-1])))

    def expand(self):
        # This function's output will be feeded to the DQN network
        origin=np.copy(self.pattern)
        # Origin pattern
        