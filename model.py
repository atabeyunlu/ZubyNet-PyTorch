import torch 
from layers import resnet_blocks
import torch.nn.functional as F
import torch.nn as nn
class ZubyNetV3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ZubyNetV3, self).__init__()
        

        self.grouped_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, groups=3)

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet_block1 = resnet_blocks(in_channels, out_channels, stride, right_arm=False)

        self.resnet_block2 = resnet_blocks(in_channels, out_channels, stride, right_arm=True)

        self.resnet_block3 = resnet_blocks(in_channels, out_channels, stride, right_arm=False)

        self.resnet_block4 = resnet_blocks(in_channels, out_channels, stride, right_arm=True)

        self.resnet_block5 = resnet_blocks(in_channels, out_channels, stride, right_arm=True)

        self.resnet_blocks = torch.nn.Sequential(self.resnet_block1, self.resnet_block2, 
                                                 self.resnet_block3, self.resnet_block4, 
                                                 self.resnet_block5)

        self.global_pooling = torch.nn.AvgPool2d(kernel_size=8, stride=1)

        self.output_layer = torch.nn.Sequential(torch.nn.Linear(3888, 1024), torch.nn.Linear(1024, 1024), 
                                                torch.nn.Linear(1024, 1024), torch.nn.Linear(1024, 3))
        
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        
        B, C, H, W = x.shape

        print("input shape: ", x.shape)
 
        out = self.grouped_conv(x)

        print("grouped conv shape: ", out.shape)

        out = self.maxpool1(out)

        print("maxpool1 shape: ", out.shape)

        out = self.conv1(out)

        print("conv1 shape: ", out.shape)

        out = self.maxpool2(out)

        print("maxpool2 shape: ", out.shape)

        out = self.bn1(out)

        print("bn1 shape: ", out.shape)

        out = self.conv2(out)

        print("conv2 shape: ", out.shape)

        out = self.resnet_blocks(out)

        print("resnet_blocks shape: ", out.shape)

        out = self.global_pooling(out)

        print("global_pooling shape: ", out.shape)

        out = out.view(B, -1)

        print("view shape: ", out.shape)

        out = self.output_layer(out)

        print("output_layer shape: ", out.shape)

        out = self.softmax(out)

        print("softmax shape: ", out.shape)

        return out

class CNNModel1(nn.Module):
    def __init__(self, fully_layer_1, fully_layer_2, drop_rate):
        super(CNNModel1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 2) # in_chan/out_chan/kernel_size
        self.bn1 = nn.BatchNorm2d(32) # norm for out
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 2)
        self.bn5 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2) # 2 times pooling
        self.drop_rate = drop_rate
        #self.dropout = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(32*15*15, fully_layer_1) # bs 32, last image 8x8
        self.fc2 = nn.Linear(fully_layer_1, fully_layer_2)
        self.fc3 = nn.Linear(fully_layer_2, 3)

    def forward(self, x):
        #print(x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        #print(x.shape)

        x = x.view(-1, 32*15*15)
        #print(x.shape)
        #x = self.dropout(F.relu(self.fc1(x))
        #x = self.dropout(F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.dropout(F.relu(self.fc1(x)), self.drop_rate)
        x = F.dropout(F.relu(self.fc2(x)), self.drop_rate)
        x = self.fc3(x)

        return x