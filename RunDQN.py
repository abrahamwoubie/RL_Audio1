# -*- coding: utf-8 -*-
from Environment import  Environment
from ExtractFeatures import Extract_Features
from GlobalVariables import GlobalVariables
from AgentDQN import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

Extract=Extract_Features

options=GlobalVariables #To access global variables from GlobalVariable.py
parameter=GlobalVariables #To access parameters from GlobalVariables.py
samples=Extract_Features #To access the member functions of the ExtractFeatures class
grid_size=GlobalVariables #To access the size of grid from Global Variables.py

env = Environment(grid_size.nRow,grid_size.nCol)
agent = DQNAgent(env)

if (options.use_samples):
    samples_goal = samples.Extract_Samples(grid_size.nRow - 1, grid_size.nCol - 1)
elif (options.use_pitch):
    samples_goal = samples.Extract_Pitch(grid_size.nRow - 1, grid_size.nCol - 1)
elif (options.use_spectrogram):
    samples_goal = samples.Extract_Spectrogram(grid_size.nRow - 1, grid_size.nCol - 1)
else:
    samples_goal = samples.Extract_Raw_Data(grid_size.nRow - 1, grid_size.nCol - 1)

for i in range(parameter.how_many_times):
    print("************************************************************************************")
    print("Iteration",i+1)
    Number_of_Iterations=[]
    Number_of_Episodes=[]
    reward_List = []
    filename = str(grid_size.nRow) + "X" + str(grid_size.nCol) + "_Experiment.txt"
    for episode in range(1,parameter.Number_of_episodes+1):
        file = open(filename, 'a')
        #done = False
        state = env.reset()
        #print("Start and Gola states at Episode {} is {} and  {}".format(episode+1, state, goal_state))
        state=Extract.Extract_Samples(state[0],state[1])
        state = np.reshape(state, [1, parameter.state_size])
        iterations=0
        Number_of_Episodes.append(episode)
        #for time in range(parameter.timesteps):
        while True:
            iterations+=1
            action = agent.act(state)
            next_state, reward, done = env.step(action,samples_goal)
            next_state = np.reshape(next_state, [1, parameter.state_size])
            agent.replay_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > parameter.batch_size:
                agent.replay(parameter.batch_size)
        Number_of_Iterations.append(iterations)
        reward_List.append(reward)
        print("episode: {}/{}, iteration: {}, reward {}".format(episode, parameter.Number_of_episodes, iterations, reward))

    #print(Number_of_Episodes)
    #print(Number_of_Iterations)
    #file.write("Episode = " + str(Number_of_Episodes))
    file.write(str(Number_of_Iterations))
    file.write('\n')
    file.close()
    #percentage_of_successful_episodes = (sum(reward_List) / parameter.Number_of_episodes) * 100
    #print("Percentage of Successful Episodes is {} {}".format(percentage_of_successful_episodes, '%'))

    fig = plt.figure()
    fig.suptitle('Q-Learning', fontsize=12)
    title=str(grid_size.nRow) + "X" + str(grid_size.nCol) + '_'+ str(i+1)
    fig.suptitle(title, fontsize=12)
    plt.plot(np.arange(len(Number_of_Episodes)), Number_of_Iterations)
    plt.ylabel('Number of Iterations')
    plt.xlabel('Episode Number')
    filename=title+'.png'
    plt.savefig(filename)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    print("************************************************************************************")
