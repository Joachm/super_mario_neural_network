from networkClass import *
import numpy as np
import pickle



dataName = '(-15,-2)'

with open('actionAndRewards'+dataName+'.pckl', 'rb') as f:
    actionAndRewards = pickle.load(f)

with open('prevStates'+dataName+'.pckl', 'rb') as f:
    prevStates = pickle.load(f)

with open('states'+dataName+'.pckl', 'rb') as f:
    states = pickle.load(f)



actions = [d[0] for d in actionAndRewards]
rewards = [d[1] for d in actionAndRewards]

print('action', actions[0])
print(actions[:10])
print(rewards[:10])


prevStates = tc.FloatTensor(prevStates)
states = tc.FloatTensor(states)

prevStates = prevStates.view(prevStates.shape[0],3,160,200).cuda()
states = states.view(states.shape[0],3,160,200).cuda()

prevStates /= 255
states /= 255

print(prevStates.shape)

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

numWorkers = 0
batchSize = 20

trainPrevS = prevStates[499:]
trainPrevA = tc.FloatTensor(actions[499:]).cuda()
trainStates = states[499:]
trainRewards = rewardsID[499:]

#trainRewardsReg = tc.FloatTensor(rewards[999:]).cuda()

trainRewards = tc.LongTensor(trainRewards).long().cuda()
print(trainRewards.shape)




testPrevS = prevStates[:500]
testPrevA = tc.FloatTensor(actions[:500]).cuda()
testStates = states[:500]
testRewards = rewardsID[:500]
#testRewardsReg = tc.FloatTensor(rewards[:1000]).cuda()
testRewards = tc.LongTensor(testRewards).long().cuda()



trainPSLoader = tc.utils.data.DataLoader(trainPrevS,
        batch_size = batchSize,
        num_workers = numWorkers)

trainStatesLoader = tc.utils.data.DataLoader(trainStates,
        batch_size = batchSize,
        num_workers = numWorkers)

trainRLoader = tc.utils.data.DataLoader(trainRewards,
        batch_size = batchSize,
        num_workers = numWorkers)

trainALoader = tc.utils.data.DataLoader(trainPrevA,
        batch_size = batchSize,
        num_workers = numWorkers)





testPSLoader = tc.utils.data.DataLoader(testPrevS,
        batch_size = batchSize,
        num_workers = numWorkers)

testStatesLoader = tc.utils.data.DataLoader(testStates,
        batch_size = batchSize,
        num_workers = numWorkers)

testRLoader = tc.utils.data.DataLoader(testRewards,
        batch_size = batchSize,
        num_workers = numWorkers)

testALoader = tc.utils.data.DataLoader(testPrevA,
        batch_size = batchSize,
        num_workers = numWorkers)




encCh = 10



model = SelfSupervisor()
model = model.cuda()

lossFunction = nn.MSELoss()
lossFunction2 = nn.CrossEntropyLoss()

optimizer = tc.optim.Adam(model.parameters(), lr=0.001)


minValLoss  = 9999

iterations = 10
print('start training')
for epoch in range(iterations):
    
    for phase in ['train','val']:
        train_loss = 0.0
        dec_loss = 0.0
        class_loss = 0.0
        acc = 0.0
        if phase == 'train':
            load1 = trainPSLoader
            load2 = trainStatesLoader
            load3 = trainRLoader
            load4 = trainALoader
            model.train(True)
        else:
            load1 = testPSLoader
            load2 = testStatesLoader
            load3 = testRLoader
            load4 = testALoader
            model.train(False)

        for inp,targetIm,targetClass,action in zip(load1,load2,load3, load4):
            
            optimizer.zero_grad()
            outputs, classes = model(inp, action)

            loss1 = lossFunction(outputs,targetIm)
            loss2 = lossFunction2(classes, targetClass)


            loss = loss1 + loss2
            
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
                

            train_loss += loss.item()*inp.size(0)
            dec_loss += loss1.item()*inp.size(0)
            class_loss += loss2.item()*inp.size(0)
        leng = len(trainPSLoader)
        train_loss = train_loss/leng
        dec = dec_loss/leng
        clas = class_loss/leng


           
        if phase == 'train':
            print("Epoch:{} \t {}Loss: {:.4f} \t decoding: {:.3f}, class: {:.3f}".format(
                epoch, phase, train_loss, dec, clas
                ))
        else:
            print("\t \t {}Loss: {:.4f} \t decoding: {:.3f}, class: {:.3f}".format(
                phase, train_loss, dec, clas
                ))
            if train_loss < minValLoss:
                minValLoss = train_loss
                tc.save(model.state_dict(), 
                        'newMarioModel2Best.model')
                print('saved new model')


tc.save(model.state_dict(), 'newMarioModel2.model')

