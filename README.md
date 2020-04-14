# super_mario_neural_network
teaching a neural network to play Super Mario

This is an ongoing project, that attempts to create a neural network, which plays Super Mario by experience.

Dependencies:
- gym_super_mario_bros from https://github.com/Kautenja/gym-super-mario-bros
- pytorch
- pickle
- numpy


The programs are implemented with the computational power and memory constriant of an average gaming laptop in mind.

firstDataCollection.py saves tuples of state_t, action_t, state_t+1, reward_t+1.
A tuple is saved for every frame encountered by random actions.
The number of saved tuples can be specified.
One can then choose to only save situations with a specific reward value, and run the data collection for each reward, to get a balanced data set.

In actionNetwork.py, a neural network is trained on the collected data.
The network takes state_t and action_t as inputs and predicts state_t+1 and reward_t+1.
The accuracy in predicting rewards on a test set  can be checked using chechRewardAccuracy.py.

When the network achieves satisfying results, the saved model can be used to play the game.
In a simple setup, the network can for each frame predict which reward each available action can lead to, and then choose the action with the highest predicted reward.
Data can then be collected from the behavior of the neural network, and the model can be retrained on this data.

The project can be improved by training a model on top of the neural network, to predict the best action_t+1 from state_1 and action_t, by using the predicted best action from the neural network as the ground truth.

