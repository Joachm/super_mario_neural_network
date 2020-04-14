from networkClass import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


tryNum = '(-15,-2)'

with open('actionAndRewards'+tryNum+'.pckl', 'rb') as f:
    actionAndRewards = pickle.load(f)

with open('prevStates'+tryNum+'.pckl', 'rb') as f:
    prevStates = pickle.load(f)

#with open('states'+tryNum+'.pckl', 'rb') as f:
#    states = pickle.load(f)


actions = [d[0] for d in actionAndRewards]
rewards = [d[1] for d in actionAndRewards]

prevStates = tc.FloatTensor(prevStates)
#states = tc.FloatTensor(states)

prevStates = prevStates.view(prevStates.shape[0],3,160,200)#.cuda()
#states = states.view(states.shape[0],3,160,200)#.cuda()

prevStates /= 255
#states /= 255


def toID(X):
    setX = list(set(X))
    IDList = []

    for x in X:
        IDList.append(setX.index(x))

    return IDList

def toOneHot(X):
    oneHots = []
    maxx = 7 #max(X)
    for x in X:
        onehot = np.zeros((1,maxx))
        onehot[0,x] = 1
        oneHots.append(onehot)
    return oneHots

actions = toOneHot(actions)
rewardsID = toID(rewards)

testPrevS = prevStates[:500].cuda()
testPrevA = tc.FloatTensor(actions[:500]).cuda()
#testStates = states[:500]
testRewards = tc.LongTensor(rewardsID[:500]).cuda()
#testRewards = testRewards#.long().cuda()

batchSize = 20
numWorkers = 0

testPSLoader = tc.utils.data.DataLoader(testPrevS,
        batch_size = batchSize,
        num_workers = numWorkers)

testALoader = tc.utils.data.DataLoader(testPrevA,
        batch_size = batchSize,
        num_workers = numWorkers)

testRLoader = tc.utils.data.DataLoader(testRewards,
        batch_size = batchSize,
        num_workers = numWorkers)

print('loading model...')
model = SelfSupervisor()
model.load_state_dict(tc.load('marioModel2Best.model'))

#model = tc.load('MarioModel2.model')

model = model.cuda()


print('making predictions...')

Acc = 0
for inp, targets, action in zip(testPSLoader, testRLoader,testALoader):
    _, classes = model(inp, action)
    
    classes = tc.argmax(classes, 1)
    evaluation = classes == targets
    evaluation = evaluation.float()

    acc = tc.sum(evaluation)/batchSize    

    Acc += acc

Acc /= len(testPSLoader)
print('accuracy', Acc.item()*100)












