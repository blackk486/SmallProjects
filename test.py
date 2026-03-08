import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder








class ResBlock(nn.Module):
    def __init__(self,in_channels,hid_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels,hid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(hid_channels),
            nn.SiLU(),
            nn.Conv2d(hid_channels,hid_channels,kernel_size=5,padding=2),
            nn.BatchNorm2d(hid_channels),
            nn.SiLU(),
            nn.Conv2d(hid_channels,in_channels,kernel_size=5,padding=2),
        )
    def forward(self,x):
        return x+self.cnn(x)
    
class Resnet(nn.Module):
    def __init__(self,in_channels,out_channels,res_channels,hid_channels,res_num):
        super().__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.hid_channels = hid_channels
        self.res_num = res_num
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,res_channels,kernel_size=3,padding=1),
            nn.SiLU()
        )
        self.decoder = nn.Conv2d(res_channels,out_channels,kernel_size=3,padding=1)
        self.hid_res_block = nn.Sequential(
            *[ResBlock(res_channels,hid_channels) for _ in range(res_num)]
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.hid_res_block(x)
        x = self.decoder(x)
        return x.mean(dim=[-2,-1])
    

x = torch.rand([1,3,256,256])
resnet = Resnet(3,20,16,32,8)
x = resnet(x)
print(x)
        

def train(model,dataloder,epoch,batch,lr):
    optim = torch.optim.Adam(model.parameters())
    loss_fun = nn.BCELoss()
    
    for i in range(epoch):
        model.train()
        for data,label in dataloder:
            optim.zero_grad()
            pred = model(data)
            loss = loss_fun(pred,label)
            loss.backward()
            print(i,loss.item())

        









