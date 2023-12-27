import torch


class resnet_blocks(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, stride, right_arm=False):
        super(resnet_blocks, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()

        if right_arm:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):

        out = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.leaky_relu(out)

        return out