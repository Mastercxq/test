import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
a = torch.randn([4, 1, 28, 28])
#1.改变维度
print(a.shape)
print(a.unsqueeze(0).shape)
# 可以把（6,28,28）变成（1,6,28,28）

#2.计算预测对的总数
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).equal(labels).sum().item()

#3.读取每一个batch
train_data=torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_data = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader=torch.utils.data.Dataloder(dataset=train_data,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=100,
                                          shuffle=False)
for epoch in range(5):
    for batch in train_loader:
        images,labels=batch
        preds=network(images)
        loss=F.cross_entropy(preds,labels)
        optim=torch.optim.Adam(network.parameters(),lr==0.01)
        optim.zero_grad()
        loss.backward()
        optim.step()