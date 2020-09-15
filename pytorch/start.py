import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 汉字均为我个人理解，英文为原文标注。
class Net(nn.Module):

    def __init__(self):
        # 继承原有模型
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # 定义了两个卷积层
        # 第一层是输入1维的（说明是单通道，灰色的图片）图片，输出6维的的卷积层（说明用到了6个卷积核，而每个卷积核是5*5的）。
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5),padding=2)
        # 第一层是输入1维的（说明是单通道，灰色的图片）图片，输出6维的的卷积层（说明用到了6个卷积核，而每个卷积核是5*5的）。
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5,5),padding=2)
        # an affine operation: y = Wx + b
        # 定义了三个全连接层，即fc1与conv2相连，将16张5*5的卷积网络一维化，并输出120个节点。
        self.fc1 = nn.Linear(16 * 8* 8, 120)
        # 将120个节点转化为84个。
        self.fc2 = nn.Linear(120, 84)
        # 将84个节点输出为10个，即有10个分类结果。
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # 用relu激活函数作为一个池化层，池化的窗口大小是2*2，这个也与上文的16*5*5的计算结果相符（一开始我没弄懂为什么fc1的输入点数是16*5*5,后来发现，这个例子是建立在lenet5上的）。
        # 这句整体的意思是，先用conv1卷积，然后激活，激活的窗口是2*2。
        print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.size())
        # If the size is a square you can only specify a single number
        # 作用同上，然后有个需要注意的地方是在窗口是正方形的时候，2的写法等同于（2，2）。
        # 这句整体的意思是，先用conv2卷积，然后激活，激活的窗口是2*2。
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        print(x.size())
        # 这句整体的意思是，调用下面的定义好的查看特征数量的函数，将我们高维的向量转化为一维。
        x = x.view(-1, self.num_flat_features(x))
        # 用一下全连接层fc1，然后做一个激活。
        x = F.relu(self.fc1(x))
        # 用一下全连接层fc2，然后做一个激活。
        x = F.relu(self.fc2(x))
        # 用一下全连接层fc3。
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 承接上文的引用，这里需要注意的是，由于pytorch只接受图片集的输入方式（原文的单词是batch）,所以第一个代表个数的维度被忽略。
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
# params = list(net.parameters())

input = torch.randn(10, 1, 32, 32)
out = net(input)


target=torch.randn(10,10)

target = target.view(1, -1)
out=out.view(1,-1)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
loss = criterion(out, target)
loss.backward()
optimizer.step()    # Does the update

