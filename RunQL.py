from Environment import  Environment
from AgentQL import Agent
import matplotlib.pyplot as plt
from GlobalVariables import  GlobalVariables

grid_size=GlobalVariables
parameter=GlobalVariables

env = Environment(grid_size.nRow,grid_size.nCol)
agent = Agent(env)

import numpy as np
# Train agent
print("\nTraining agent...\n")

Number_of_Iterations=[]
Number_of_Episodes=[]

for episode in range(parameter.Number_of_episodes):

    # Generate an episode
    iter_episode, reward_episode = 0, 0
    state = env.reset()  # starting state
    Number_of_Episodes.append(episode)
    iteration=0
    while True:
        iteration+=1
        action = agent.get_action(env)  # get action

        state_next, reward, done = env.step(action)  # evolve state by action
        agent.train((state, action, state_next, reward, done))  # train agent

        #state_sample, reward, done = env.step(action)  # evolve state by action
        #agent.train((state, action, state_sample, reward, done))  # train agent

        iter_episode += 1
        reward_episode += reward
        if done:
            break
        state = state_next  # transition to next state
        #state = state_sample

    Number_of_Iterations.append(iteration)
    # Decay agent exploration parameter
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

    # Print
    # if (episode == 0) or (episode + 1) % 10 == 0:
    print("[episode {}/{}], iter = {}, reward = {:.1F}".format(
        episode + 1, parameter.Number_of_episodes, iter_episode, reward_episode))

fig = plt.figure()
fig.suptitle('Q-Learning', fontsize=12)
title="Q-Learning "+str(grid_size.nRow) + "X" + str(grid_size.nCol)
fig.suptitle(title, fontsize=12)
plt.plot(np.arange(len(Number_of_Episodes)), Number_of_Iterations)
plt.ylabel('Number of Iterations')
plt.xlabel('Episode Number')
#plt.savefig('4X4.png')
plt.show()
