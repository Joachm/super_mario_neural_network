from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import numpy as np
import pickle

dataName = '(-15,-2)'

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#SIMPLE_MOVEMENT: 7 actions
#RIGHT_ONLY: 5 actions
#COMPLEX_MOVEMENT: 12 actions


print(env.get_action_meanings())


L = 30000
num = 2000

prevStates = np.zeros((num,160,200,3))
states = np.zeros((num, 160,200,3))

prevState = ''
prevAction = ''

actionAndRewards = []

done = True
count = 0
i = 0
while len(actionAndRewards) < num-1:
    if i%1000==0:
        print(i, 'frames played')
    if done:
        state = env.reset()
    action = np.random.choice(range(5))
    state, reward, done, info = env.step(action)
    #env.render()
    
    if i!=0:
        if reward < -1:
            actionAndRewards.append((prevAction, reward))
            prevStates[count] = prevState[50:-30,30:-26:]
            states[count] = state[50:-30,30:-26:]
            count += 1
    prevState = state
    prevAction = action
    i +=1
env.close()

print(len(actionAndRewards))
pickle.dump(actionAndRewards, open('actionAndRewards'+dataName+'.pckl', 'wb'))
pickle.dump(prevStates, open('prevStates'+dataName+'.pckl', 'wb'))
pickle.dump(states, open('states'+dataName+'.pckl', 'wb'))


