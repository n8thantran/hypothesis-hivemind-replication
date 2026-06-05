"""
ConvNet architecture from DCBench / DD literature.
ConvNet-D3 = 3 convolutional blocks, each with Conv-IN-ReLU-AvgPool.
Standard for CIFAR-100 evaluation in dataset distillation papers.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 norm='instancenorm', relu='relu', pool='avgpool'):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        
        if norm == 'instancenorm':
            layers.append(nn.GroupNorm(out_channels, out_channels, affine=True))
        elif norm == 'batchnorm':
            layers.append(nn.BatchNorm2d(out_channels))
        
        if relu == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif relu == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        
        if pool == 'avgpool':
            layers.append(nn.AvgPool2d(2))
        elif pool == 'maxpool':
            layers.append(nn.MaxPool2d(2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class ConvNet(nn.Module):
    """
    ConvNet-D{depth} architecture.
    For CIFAR-100 (32x32): D3 means 3 conv blocks.
    Each block: Conv(128) -> InstanceNorm -> ReLU -> AvgPool(2x2)
    After 3 blocks: 32->16->8->4, then flatten and linear.
    """
    def __init__(self, num_classes=100, channel=3, im_size=(32, 32), 
                 net_width=128, net_depth=3, net_norm='instancenorm',
                 net_act='relu', net_pooling='avgpool'):
        super().__init__()
        
        self.features = nn.Sequential()
        in_channels = channel
        for i in range(net_depth):
            self.features.add_module(f'block{i}', 
                ConvBlock(in_channels, net_width, norm=net_norm, 
                         relu=net_act, pool=net_pooling))
            in_channels = net_width
        
        # Calculate feature size after pooling
        # Each avgpool halves spatial dims
        feat_size = im_size[0] // (2 ** net_depth)
        self.classifier = nn.Linear(net_width * feat_size * feat_size, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def embed(self, x):
        """Return features before classifier."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def get_convnet_d3(num_classes=100, channel=3, im_size=(32, 32)):
    """Get ConvNet-D3 for CIFAR-100."""
    return ConvNet(num_classes=num_classes, channel=channel, im_size=im_size,
                   net_width=128, net_depth=3)


if __name__ == '__main__':
    model = get_convnet_d3()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"ConvNet-D3 output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
