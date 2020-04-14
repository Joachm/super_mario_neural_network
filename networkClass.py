import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class SelfSupervisor(nn.Module):

    def __init__(self, batchSize=20):
        super(SelfSupervisor, self).__init__()
        
        self.batchSize = batchSize
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,3,3,padding=1)


        self.lin = nn.Linear((3+1)*20*25,10)

        self.pool = nn.MaxPool2d(2,2)

        self.t_conv1 = nn.ConvTranspose2d(3+1,32,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16,3,2,stride=2)

        self.action1 = nn.Linear(7,20)
        self.action2 = nn.ConvTranspose2d(1,1,4,stride=1)

        self.prelu = nn.PReLU()

    def forward(self, x, a):
        #encode
        e = self.prelu(self.conv1(x))
        e = self.pool(e)
        e1 = self.prelu(self.conv2(e))
        e = self.pool(e1)
        e = self.prelu(self.conv3(e))
        #e = self.m(self.conv3(e))

        ##action processing
        a = self.prelu(self.action1(a))

        a = a.view(self.batchSize,1,1,20)
        a = self.action2(a)
        a = F.interpolate(a,size=(20,25))

        ##latent space
        z = self.pool(e)
        z = tc.cat((z,a),1)
        #print(z.shape)

        #reward prediction
        c = z.view(-1,(3+1)*20*25)
        classes = self.lin(c) 

        #decode
        d = self.prelu(self.t_conv1(z))
        d = self.prelu(self.t_conv2(d))
        #d = tc.cat((d,e1),1)
        d = tc.sigmoid(self.t_conv3(d))
        return d, classes



