import torch
import torch.nn as nn
import torch.optim as optim



class   P_net(nn.Module):
    def __init__(self):
        super(P_net,self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,10,kernel_size=3,stride=1),#conv1
            nn.PReLU(),#PReLU1
            nn.MaxPool2d(kernel_size=3,stride=2),#pool1
            nn.Conv2d(10,16,kernel_size=3,stride=1),#conv2
            nn.PReLU(),#PReLU2
            nn.Conv2d(16,32,kernel_size=3,stride=1),
            nn.PReLU()#PReLU3
        )
        self.Conv4_1 = nn.Conv2d(32,1,kernel_size=3,stride=1)
        self.Conv4_2 = nn.Conv2d(32,4,kernel_size=3,stride=1)


    def forward(self,x):
        x = self.pre_layer(x)
        conf = nn.Sigmoid(self.Conv4_1(x))#置信度需要用sigmoid函数激活一下在输出
        offset = self.Conv4_2(x)#偏移量直接输出
        return conf,offset



class R_net(nn.Module):
    def __init__(self):
        super(R_net,self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(24,28,kernel_size=3,stride=1),#conv1
            nn.PReLU(),#ReLU1
            nn.MaxPool2d(kernel_size=3,stride=2),#pool1
            nn.Conv2d(28,48,kernel_size=3,stride=1),#conv2
            nn.PReLU(),#ReLU2
            nn.MaxPool2d(kernel_size=3,stride=2),#pool2
            nn.Conv2d(48,64,kernel_size=2,stride=1),#conv3
            nn.PReLU(),#ReLU3



        )
        self.Conv4 = nn.Linear(64*2*2,128)#conv4
        self.PReLU4 = nn.PReLU()#prelu4
        #detection
        self.Conv5_1 = nn.Linear(128,1)
        self.Conv5_2 = nn.Linear(128, 4)



    def  forward(self,x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)#传入全连接层之前要进行数据变换，nchw变成nv
        x = self.Conv4(x)
        x = self.PReLU4(x)
        conf = nn.Sigmoid(self.Conv5_1(x))
        offset = self.Conv5_2(x)
        return conf,offset


class O_net(nn.Module):
    def __init__(self):
        super(O_net,self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(48,32,kernel_size=3,stride=1),#conv1
            nn.PReLU(),#RELU1
            nn.MaxPool2d(kernel_size=3,stride=2),#pool1
            nn.Conv2d(32,64,kernel_size=3,stride=1),#conv2
            nn.PReLU(),#relu2
            nn.MaxPool2d(kernel_size=3,stride=2),#pool2
            nn.Conv2d(64,64,kernel_size=3,stride=1),#conv3
            nn.PReLU(),#relu3
            nn.MaxPool2d(kernel_size=2,stride=2),#pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1),#conv4
            nn.PReLU()#relu4


        )
        self.conv5 = nn.Linear(128*2*2,256)
        self.ReLU5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256,1)
        self.conv6_2 = nn.Linear(256,4)


    def forward(self,x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        x = self.conv5(x)
        x = self.ReLU5(x)

        conf = nn.Sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return conf,offset