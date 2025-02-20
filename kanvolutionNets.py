import torch
from torch import nn
from torch.nn import functional as F
from custom_layer import Kanvolution2d

__all__ = ['KanvolutionalNetwork', 'KanVolNet']


class KanvolutionLayer(nn.Module):
    """
    This layer is an approximation of a kanvolutional layer with kernel size 1 x 1
    """
    def __init__(self, in_channels, out_channels, order_p:int=5, order_q:int=4, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, device='cuda'):
        super(KanvolutionLayer, self).__init__()
        self.p = order_p
        self.q = order_q
        self.device = device

        
        # init the internal phi_ij kernel functions
        setattr(self, 'convp_1', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True).to(device))
        for i in range(2, self.p+1):
            setattr(self, f'convp_{i}', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False).to(device))
        for i in range(1, self.q+1):
            setattr(self, f'convq_{i}', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False).to(device))
        # init bias for inner sum
        self.bias = nn.Parameter(torch.randn((*kernel_size, out_channels))).to(device)


    def forward(self, x):
        
        sum_p = sum(getattr(self, f'convp_{i}')(x**i) for i in range(1, self.p+1))
        sum_q = torch.zeros(1).to(self.device)
        if self.q != 0:
             sum_q = sum([getattr(self, f'convq_{i}')(x**i) for i in range(1, self.q+1)])

        return (sum_p / (1 + torch.abs(sum_q)))
    

class BlockKanvolution(nn.Module):
    def __init__(self, in_channels, out_channels, order_p:int=5, order_q:int=4 ,kernel_size=(5,5), stride=1, padding=0, dilation=1, groups=1, device='cuda'):
        super(BlockKanvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3,3), stride, padding, dilation, groups, bias=True).to(device)
        self.kanv = KanvolutionLayer(out_channels, 4*out_channels, order_p, order_q, kernel_size, stride, padding, dilation, groups, device)
        self.batchnorm = nn.BatchNorm2d(out_channels*4).to(device)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.kanv(x)
        x = self.batchnorm(x)
        return x
    
class KanvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, order_p:int=5, order_q:int=4 ,kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, device='cuda'):
        super(KanvolutionalNetwork, self).__init__()
        self.block1 = BlockKanvolution(in_channels, out_channels, order_p, order_q, kernel_size, stride, padding, dilation, groups, device)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.flatten = nn.Flatten().to(device)
        self.fc1 = nn.LazyLinear(10).to(device)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    

class KanVolNet(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            order_p:int=5, 
            order_q:int=4,
            kernel_size=(3,3), 
            stride=1, 
            padding=0,
            bias:bool=True,
            scale:bool=False, 
            device:str='cuda'
            ):
        super(KanVolNet, self).__init__()
        self.conv1 = Kanvolution2d(in_channels=in_channels, out_channels=out_channels, order_p=order_p, order_q=order_q, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, scale=scale)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.batch1 = nn.BatchNorm2d(out_channels).to(device)
        self.conv2 = Kanvolution2d(in_channels=out_channels, out_channels=2*out_channels, order_p=order_p, order_q=order_q, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, scale=scale)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.batch2 = nn.BatchNorm2d(2*out_channels).to(device)
        self.flatten = nn.Flatten().to(device)
        self.fc1 = nn.LazyLinear(10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.batch2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x