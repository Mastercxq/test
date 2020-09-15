import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(1,16,kernel_size=(5,5),stride=1,padding=2),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(16,32,kernel_size=(5,5),stride=1,padding=2),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc=nn.Linear(7*7*32,num_classes)#这里的7*7是怎么算出来的呢:第一层输入图片大小为28×28，（5,5）卷积核，
        # 填充两层0为32×32，步长为1，得到的输出图像边长大小为32-5+1=28，经过（2,2）池化核和2步长后大小为14×14
        #下一步同上，大小最后为7×7

    def forward(self,x):
        out=self.layer1(x)
        # print(out.size())
        out=self.layer2(out)
        # print(out.size())
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

model=Net(num_classes)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
optimizer.zero_grad()
total_step=len(train_loader)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        # print(images.size())
        labels = labels.to(device)

        outputs = model(images)
        # print(outputs.size())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))