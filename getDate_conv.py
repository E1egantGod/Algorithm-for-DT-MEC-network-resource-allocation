import torch
from torch import nn
import numpy as np
import time as t
from torch.autograd import Variable
import random
import xlwt

class Convlayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, i):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride),
        )
            
    def forward(self, x):
        t1 = t.time()
        x = self.conv(x)
        t2 = t.time()
        sheet.write(i,0,t2-t1)
        return x  

i_max = 50000
wb = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = wb.add_sheet('Sheet1',cell_overwrite_ok=True)

print("开始获取数据")
for i in range(i_max):
    in_ch = random.randint(1,512)
    out_ch = random.randint(1,512)
    kernel_size = random.randint(1,8)
    stride = random.randint(1,4)
    in_size = random.randint(10,227)

    conv = Convlayer(in_ch,out_ch,kernel_size,stride,i)

    input_data = Variable(torch.rand(1,in_ch,in_size,in_size))
    output = conv(input_data)

    sheet.write(i,1,in_ch)
    sheet.write(i,2,out_ch)
    sheet.write(i,3,kernel_size)
    sheet.write(i,4,output.size()[2])
    sheet.write(i,5,stride)
    print("已生成",i,"组数据", flush=True)

wb.save('data_conv.xls')
print("结束获取数据")