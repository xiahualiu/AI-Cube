from cube import cube
from model import RL
from copy import deepcopy

class MCTS:
    def __init__(self, cube, net):
        self.root=node(cube, net, [])
        self.shortest=float('inf')
        self.best_path=[]
        self.best_node=self.root # find the potential best node in the tree
        self.max_search=6*5*5*5;
        self.search_cnt=0;

    def search(self):
        path=[]
        queue=[self.root]
        self.search_cnt=0;

        while len(queue)!=0 and self.search_cnt < self.max_search:
            node = queue.pop(0)
            self.search_cnt+=1
            if node.isleaf:
                if node.value > self.best_node.value:
                    self.best_node=node
                if node.solvable:
                    temp_path=node.lastactions+node.path
                    if len(temp_path)<self.shortest:
                        self.best_path=temp_path
                        self.shortest=len(temp_path)
            else:
                queue=queue+node.child

    def add_leaves(self):
        path=[]
        queue=[self.root]
        while queue:
            node = queue.pop(0)
            if (node.isleaf) and (not node.issolved):
                node.add_child()
            else:
                queue=queue+node.child

class node():
    def __init__(self, cube, net, lastactions):
        self.cube=cube
        self.net=net
        self.issolved=cube.check(cube.state)
        self.lastactions=lastactions
        self.isleaf=True
        self.child=[]
        # Below properties are going to be written by simulate function
        self.solvable=False
        self.path=[]
        self.distance=float('inf')
        self.value=0
        # Do simulate for every new node
        self.simulate()

    def add_child(self):
        actions=[0,1,2,3,4,5]
        if len(self.lastactions) != 0:
            actions.remove(self.get_action(self.lastactions[-1]))
        for action in actions:
            self.child.append(self.turn_node(action))
        self.isleaf=False
        print

    def turn_node(self, action):
        temp_cube=self.cube.new_cube(action)
        temp_node=node(temp_cube,self.net,self.lastactions+[action])
        return temp_node

    def get_action(self,action):
        if action<3:
            counter=action+3
        else:
            counter=action-3
        return counter

    def simulate(self):

        # Copy the cube and do the simulation
        temp_cube = self.cube.copy()

        for i in range(10):
            # Check if the cube can be solved in 10 steps
            # when it >10 steps we regard it is not solvable by our network
            if temp_cube.check(temp_cube.state):
                self.solvable=True
                self.distance=i
                return

            policy_predict, value_predict = self.net.predict_state(temp_cube.state)
            self.value=value_predict.detach().numpy()
            if len(self.lastactions) != 0:
                policy_predict[self.get_action(self.lastactions[-1])]=-float('inf')
            # By setting it to -inf we make sure not to select it on next line
            _, max_act_t = policy_predict.max(dim=0)
            action=max_act_t.detach().numpy()
            last_move=action
            temp_cube.turn(action)
            self.path.append(action)
            # If 5 steps did not solve the cube
        self.solvable=False
        self.distance=float('inf')
        self.path=[]
        return