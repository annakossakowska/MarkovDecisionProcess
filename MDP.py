import numpy as np


#Multiplier is R(s,a,s')+gamma*V_in(s')
#V_in is the V from the previous loop
def make_multiplier(V_in_vector, reward_vector, gamma_value):
    a = np.roll(V_in_vector, 1)
    b = np.roll(V_in_vector, 0)
    c = np.roll(V_in_vector, -1)
    rolled_V_in = np.column_stack((a, b, c))
    multiplier = reward_vector[:, np.newaxis] + gamma * rolled_V_in
    return multiplier

def calculate_new_V(T, multiplier):
    triplet = np.sum(T * multiplier[:, np.newaxis, :], axis=2)
    V_out = np.max(triplet, axis=1)
    return V_out

def run(V_in,reward,gamma,T,K):
    for i in range(K):
        multiplier = make_multiplier(V_in, reward, gamma)
        V_in = calculate_new_V(T, multiplier)
    return V_in


#Defining the matrix with probabilities for each situation. Matrix 5x3x3
#There is 5 positions, each have three choices, and each choice can have three effects with different probabilities.
T = np.array([[[0,1/2,1/2],[0, 1/2, 1/2],[0, 2/3, 1/3]],[[1/3,2/3,0],[1/4,1/2,1/4],[0,2/3,1/3]],[[1/3,2/3,0],[1/4,1/2,1/4],[0,2/3,1/3]],[[1/3,2/3,0],[1/4,1/2,1/4],[0,2/3,1/3]],[[1/3,2/3,0],[1/2,1/2,0],[1/2,1/2,0]]])

#each position has its own reward, which doeasn't change
reward = np.array([0,0,0,0,1])

#Initialization V input
V_in = np.array([0,0,0,0,0])

K=10
gamma=0.5

print("Value iteration after %d iterations: " %(K),run(V_in,reward,gamma,T,K))