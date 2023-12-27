import torch 
from layers import resnet_blocks

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