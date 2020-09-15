import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,10,5)
        self.conv2=nn.Conv2d(10,50,3)
        self.maxpool2D=nn.MaxPool2d((2,2),2)
        self.fc1=nn.Linear(50,60)
        self.fc2=nn.Linear(60,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.flatten(x))
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    def flatten(self,x):
        size=x.size()[1:]
        numfeature=1
        for i in size:
            numfeature*=i
        return numfeature

net=Net()
print(net)