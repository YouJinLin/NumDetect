import torch
from torch.utils import data as Data
import torch.nn as nn
import torchvision
import os

EPOCH = 8
BATCH_SIZE = 50  #批量訓練50組資料
LR = 0.001
DOWNLOAD_MNIST = False

if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST
)

print(f'Train_Data:{train_data.train_data.size()}')
print(f'Train_Label:{train_data.train_labels.size()}')

train_load = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:1000]/255.
test_y = test_data.test_labels[:1000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding =  2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_load):
        out = cnn(b_x)[0]
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 追蹤訓練過程
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)  #輸入資料已轉換為概率(0~1)
            pred_y = torch.max(test_output, 1)[1].data.numpy()  #輸出資料亦為概率,所以不須F.softmax處理
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum())  / float(test_y.size(0))
            print(f'epoch:{epoch}, train_loss:{loss.data.numpy()}, test_accuracy:{accuracy}')
# 比對判斷結果
test_output, _ = cnn(test_x[11:20])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[11:20].numpy(), 'real number')

#保存模型
try:
    torch.save(cnn, r'D:/python/Numbers/numbers.pth')
    print('save success')
except:
    print('save failure')
