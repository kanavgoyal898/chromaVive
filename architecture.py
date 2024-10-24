import torch
import torch.nn as nn

from normalizer import *

class Architecture(Normalizer):

    def __init__(self, bin_count=313, out_channels=64, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        self.bin_count = bin_count
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # convolution1: (256, 256, 1) -> (128, 128, 64)
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels)
        )

        # convolution2: (128, 128, 64) -> (64, 64, 128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels*2)
        )
        self.out_channels = self.out_channels * 2

        # convolution3: (64, 64, 128) -> (32, 32, 256)
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels*2)
        )
        self.out_channels = self.out_channels * 2

        # convolution4: (32, 32, 256) -> (32, 32, 512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels*2)
        )
        self.out_channels = self.out_channels * 2

        # a-trous / dilated convolution1: (32, 32, 512) - > (32, 32, 512)
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding*2, dilation=self.stride*2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding*2, dilation=self.stride*2, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels),
        )
        self.out_channels = self.out_channels * 1

        # a-trous / dilated convolution2: (32, 32, 512) - > (32, 32, 512)
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.stride*2, dilation=self.stride*2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.stride*2, dilation=self.stride*2, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels)
        )
        self.out_channels = self.out_channels * 1

        # convolution5: (32, 32, 512) - > (32, 32, 512)
        self.conv7 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels)
        )
        self.out_channels = self.out_channels * 1

        # inverse-convolution: (32, 32, 512) -> (64, 64, 256)
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels//2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels//2, self.out_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels//2, self.out_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels//2, self.bin_count, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.out_channels = self.out_channels // 2

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model = Architecture()
model = model.to(device=device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_parameters:,} parameters on device: {device}')
