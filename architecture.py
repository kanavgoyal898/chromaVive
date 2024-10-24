import torch
import torch.nn as nn

from normalizer import *

class Architecture(Normalizer):
    """
    Initializes the Architecture model with the specified parameters.

    Args:
        bin_count (int): The number of output classes or bins (default: 313).
        in_channel_count (int): The number of input channels for the first convolution layer (default: 1).
        out_channel_count (int): The number of output channels for the final layer (default: 2).
        scale_factor (int): Factor to upsample the output (default: 4).
        out_channels (int): Initial number of output channels for the convolution layers (default: 64).
        kernel_size (int): Size of the convolution kernel (default: 3).
        stride (int): Stride of the convolution (default: 1).
        padding (int): Padding for the convolution (default: 1).
        norm_layer (callable): Normalization layer to use (default: nn.BatchNorm2d).
    """

    def __init__(self, bin_count=313, in_channel_count=1, out_channel_count=2, scale_factor=4, out_channels=64, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        self.bin_count = bin_count
        self.in_channel_count = in_channel_count
        self.out_channel_count = out_channel_count
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # convolution1: (256, 256, 1) -> (128, 128, 64)
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            # expansion layer
            nn.Conv2d(self.in_channel_count, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            # down-sampling layer
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels)
        )

        # convolution2: (128, 128, 64) -> (64, 64, 128)
        self.conv2 = nn.Sequential(
            # expansion layer
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            # down-sampling layer
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels*2)
        )
        self.out_channels = self.out_channels * 2

        # convolution3: (64, 64, 128) -> (32, 32, 256)
        self.conv3 = nn.Sequential(
            # expansion layer
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            # down-sampling layer
            nn.Conv2d(self.out_channels*2, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            norm_layer(self.out_channels*2)
        )
        self.out_channels = self.out_channels * 2

        # convolution4: (32, 32, 256) -> (32, 32, 512)
        self.conv4 = nn.Sequential(
            # expansion layer
            nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            # down-sampling layer
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
            # contraction layer
            nn.ConvTranspose2d(self.out_channels, self.out_channels//2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels//2, self.out_channels//2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels//2, self.out_channels//2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True),
            nn.ReLU(inplace=True),
            # custom deconvolution layer
            nn.ConvTranspose2d(self.out_channels//2, self.bin_count, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.out_channels = self.out_channels // 2

        self.out = nn.Conv2d(self.bin_count, self.out_channel_count, kernel_size=1, stride=1, padding=0, bias=True)

        # Bilinear interpolation considers the nearest four pixel values (the surrounding 2x2 grid) and computes the output pixel value based on a weighted average of these pixels. 
        # This results in smoother and more visually appealing upsampled images compared to other methods, like nearest-neighbor interpolation.
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')

    def forward(self, x):
        """
        Defines the forward pass of the Architecture model.

        Args:
            x (torch.Tensor): Input tensor with shape 
            (batch_size, in_channel_count, height, width, ).

        Returns:
            torch.Tensor: Output tensor after processing through the network, 
                        with shape (batch_size, out_channel_count, upsampled_height, 
                        upsampled_width) after upsampling.
        """

        x = self.normalize_l(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = nn.functional.softmax(x, dim=1)
        x = self.out(x)
        x = self.upsample(x)

        return self.denormalize_ab(x)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model = Architecture()
model = model.to(device=device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_parameters:,} parameters on device: {device}')

x = torch.randn(8, 1, 256, 256).to(device=device)
output = model(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {output.shape}')