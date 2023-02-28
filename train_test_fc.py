import torch
from torch import nn
import numpy as np
import time as t
from torch.autograd import Variable
import xlrd
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 1)
        )

        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  

wb = xlrd.open_workbook('data_fc.xls')
sheet = wb.sheet_by_name('Sheet1')
i_train = 28000
i_test = 12000
net = Net()
LR = 1e-3
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=LR)

losss = []
print("开始训练")
for i in range(i_train):
    cells = sheet.row_values(i)
    # print(cells)
    input = torch.tensor(cells[1:],dtype=torch.float).reshape(1,1,2)
    output = net(input)
    # print(output)
    loss = loss_fn(output,torch.tensor(cells[0],dtype=torch.float).reshape(1,1))
    losss.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%10000 == 0:
        print("已经训练",i+1,"步")
print('结束训练')

outputs = []
rights = []
accs = []
acc_counter = 0
print("开始测试")
for j in range(i_test):
    cells = sheet.row_values(int(j+i_train))
    # cells = sheet.row_values(int(j))
    input = torch.tensor(cells[1:],dtype=torch.float).reshape(1,1,2)
    output = net(input)
    outputs.append(output.detach().numpy()[0])
    rights.append(cells[0])
    error = output.detach().numpy()[0] - cells[0]
    if -0.1 * cells[0] <= error and 0.1 * cells[0] >= error:
        acc_counter = acc_counter + 1

print("acc: ", acc_counter/i_test)
