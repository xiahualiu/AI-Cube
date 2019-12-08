import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RL

def MCTS(mycube):
    """
    Will start Monte Carlo Tree Search propagating 3 turns out and finding the most optimal starting move
    Input: Cube Object
    Output: Solution vector containing turns needed to reach solution, new cube object with first optimal action
    """
    set_search_actions=[
    [0,0,0]
    [0,0,1]
    [0,0,2]
    [0,0,4]
    [0,0,5] 
    [0,1,0]
    [0,1,1]
    [0,1,2]
    [0,1,3]
    [0,1,5]
    [0,2,0]
    [0,2,1]
    [0,2,2]
    [0,2,3]
    [0,2,4]
    [0,4,0]
    [0,4,2]
    [0,4,3]
    [0,4,4]
    [0,4,5]
    [0,5,0]
    [0,5,1]
    [0,5,3]
    [0,5,4]
    [0,5,5]
    [1,0,0]
    [1,0,1]
    [1,0,2]
    [1,0,4]
    [1,0,5] 
    [1,1,0]
    [1,1,1]
    [1,1,2]
    [1,1,3]
    [1,1,5]
    [1,2,0]
    [1,2,1]
    [1,2,2]
    [1,2,3]
    [1,2,4]
    [1,3,1]
    [1,3,2]
    [1,3,3]
    [1,3,4]
    [1,3,5]
    [1,5,0]
    [1,5,1]
    [1,5,3]
    [1,5,4]
    [1,5,5]
    [2,0,0]
    [2,0,1]
    [2,0,2]
    [2,0,4]
    [2,0,5] 
    [2,1,0]
    [2,1,1]
    [2,1,2]
    [2,1,3]
    [2,1,5]
    [2,2,0]
    [2,2,1]
    [2,2,2]
    [2,2,3]
    [2,2,4]
    [2,3,1]
    [2,3,2]
    [2,3,3]
    [2,3,4]
    [2,3,5]
    [2,4,0]
    [2,4,2]
    [2,4,3]
    [2,4,4]
    [2,4,5]
    [3,1,0]
    [3,1,1]
    [3,1,2]
    [3,1,3]
    [3,1,5]
    [3,2,0]
    [3,2,1]
    [3,2,2]
    [3,2,3]
    [3,2,4]
    [3,3,1]
    [3,3,2]
    [3,3,3]
    [3,3,4]
    [3,3,5]
    [3,4,0]
    [3,4,2]
    [3,4,3]
    [3,4,4]
    [3,4,5]
    [3,5,0]
    [3,5,1]
    [3,5,3]
    [3,5,4]
    [3,5,5]
    [4,0,0]
    [4,0,1]
    [4,0,2]
    [4,0,4]
    [4,0,5] 
    [4,1,0]
    [4,1,1]
    [4,1,2]
    [4,1,3]
    [4,1,5]
    [4,3,1]
    [4,3,2]
    [4,3,3]
    [4,3,4]
    [4,3,5]
    [4,4,0]
    [4,4,2]
    [4,4,3]
    [4,4,4]
    [4,4,5]
    [4,5,0]
    [4,5,1]
    [4,5,3]
    [4,5,4]
    [4,5,5]
    [5,0,0]
    [5,0,1]
    [5,0,2]
    [5,0,4]
    [5,0,5] 
    [5,1,0]
    [5,1,1]
    [5,1,2]
    [5,1,3]
    [5,1,5]
    [5,3,1]
    [5,3,2]
    [5,3,3]
    [5,3,4]
    [5,3,5]
    [5,4,0]
    [5,4,2]
    [5,4,3]
    [5,4,4]
    [5,4,5]
    [5,5,0]
    [5,5,1]
    [5,5,3]
    [5,5,4]
    [5,5,5]
    ]
    #This vector is a map of all moves extending out to three turns. I know this is probably not the most optimal way to do this
    #but it the only way I could think of
    state_values_for_action=np.zeros[150,3] #create empty matrix to store move values
    i=range(149)
    for x in i:  
        temp_cube=mycube.copy() #create temporary copy of the current cube
        temp_cube=temp_cube.turn(set_search_actions[0][x]) # apply (from set_search_actions) the first move
        state_values_for_action[0][x]=predict_cube(temp_cube) #record value for this cube state in state_value_for_action
        temp_cube=temp_cube.turn(set_search_actions[1][x]) # apply (from set_search_actions) the second move
        state_values_for_action[1][x]=predict_cube(temp_cube) #record value for this cube state in state_value_for_action
        temp_cube=temp_cube.turn(set_search_actions[2][x]) # apply (from set_search_actions) the third move
        state_values_for_action[2][x]=predict_cube(temp_cube) #record value for this cube state in state_value_for_action
    max_value_of_search=max(state_value_for_action) #locate maximum value position from the 186 states recorded in state_values_for_action
    solution_pos=np.where(state_value_for_action==max_value_of_search) #find the index (position) of the maximum value in state_values_for_action
    solution=solution.append(set_search_actions[solution_pos[0]][0]) #append the first turn that led to the maximum value
    mycube=mycube.turn(solution_num) #update the solution vector to include the turn
    return solution, mycube #output the updated solution vector and the new mycube object
    