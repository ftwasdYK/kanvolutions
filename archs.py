from PIL import Image
import os
import torch
Ανάπτυξη
utils_training.py
4 KB
﻿
giannisK
giannisk_34682
from torch import nn
import torch
from custom_layer import Kanvolution2d

__all__ = ['UNet', 'UKonvNet']

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer:nn.Module=nn.Conv2d):
        super(ResidualBlock, self).__init__()
        

        self.conv1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu_flag = isinstance(self.conv1, nn.Conv2d)

        if self.relu_flag:
            self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.relu_flag:
            out = self.relu(out)
        
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        residual = self.conv1(residual)
        residual = self.bn3(residual)

        out += residual
        if self.relu_flag:
            out = self.relu(out)

        return out
    

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer:nn.Module=nn.Conv2d):
        super(encoder_block, self).__init__()
        self.conv1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_flag = isinstance(self.conv1, nn.Conv2d)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if self.relu_flag:
            self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu_flag:
            x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        if self.relu_flag:
            x = self.relu(x)
        
        next_layer = self.max_pool(x)
        skip_layer = x
        
        return next_layer, skip_layer
    
class res_encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer:nn.Module=nn.Conv2d):
        super(res_encoder_block, self).__init__()
        

        self.conv1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_flag = isinstance(self.conv1, nn.Conv2d)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if self.relu_flag:
            self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu_flag:
            x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        residual = self.conv1(residual)
        residual = self.bn3(residual)
        
        
        x += residual  
        if self.relu_flag:
            x = self.relu(x)
        
        next_layer = self.max_pool(x)
        skip_layer = x
        
        return next_layer, skip_layer
    
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer:nn.Module=nn.Conv2d):
        super(decoder_block, self).__init__()
        
        
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv1 = conv_layer(in_channels = 2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_flag = isinstance(self.conv1, nn.Conv2d)        
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.relu_flag:
            self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x, skip_layer):
        x = self.transpose_conv(x)
        x = torch.cat([x, skip_layer], axis=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu_flag:
            x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        if self.relu_flag:
            x = self.relu(x)
        
        return x
    

class res_decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer:nn.Module=nn.Conv2d):
        super(res_decoder_block, self).__init__()
        
        
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv1 = conv_layer(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_flag = isinstance(self.conv1, nn.Conv2d)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if self.relu_flag:
            self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x, skip_layer):
        x = self.transpose_conv(x)
        x = torch.cat([x, skip_layer], axis=1)
        
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu_flag:
            x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        residual = self.conv1(residual)
        residual = self.bn3(residual)
        
        x += residual
        if self.relu_flag:
            x = self.relu(x)
        
        return x
    
class bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer:nn.Module=nn.Conv2d):
        super(bottleneck_block, self).__init__()
        

        self.conv1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_flag = isinstance(self.conv1, nn.Conv2d)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.relu_flag:
            self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.relu_flag:
            x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        if self.relu_flag:
            x = self.relu(x)
        
        return x
    


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, layers_stack:tuple=(64, 128, 256, 512)):
        super(UNet, self).__init__()
        # Encoder blocks
        self.enc1 = encoder_block(in_channels, layers_stack[0], conv_layer=nn.Conv2d)
        self.enc2 = encoder_block(layers_stack[0], 2*layers_stack[0])
        self.enc3 = res_encoder_block(layers_stack[1], 2*layers_stack[1])
        self.enc4 = encoder_block(layers_stack[2], 2*layers_stack[2])
        
        # Bottleneck block
        self.bottleneck = ResidualBlock(layers_stack[3], 2*layers_stack[3]) # 512 to 1024
        
        # Decoder blocks
        self.dec1 = decoder_block(2 * layers_stack[3], layers_stack[3])
        self.dec2 = res_decoder_block(2 * layers_stack[2], layers_stack[2])
        self.dec3 = decoder_block(2 * layers_stack[1] , layers_stack[1])
        self.dec4 = decoder_block(2 * layers_stack[0], layers_stack[0])
        
        # 1x1 convolution
        self.out = nn.Conv2d(layers_stack[0], num_classes, kernel_size=1, padding='same')
        
    def forward(self, image):
        n1, s1 = self.enc1(image)
        n2, s2 = self.enc2(n1)
        n3, s3 = self.enc3(n2)
        n4, s4 = self.enc4(n3)
        
        n5 = self.bottleneck(n4)
        
        n6 = self.dec1(n5, s4)
        n7 = self.dec2(n6, s3)
        n8 = self.dec3(n7, s2)
        n9 = self.dec4(n8, s1)
        
        output = self.out(n9)
        
        return output
        
class UKonvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, layers_stack:tuple=(64, 128, 256, 512)):
        super(UKonvNet, self).__init__()
        # Encoder blocks
        self.enc1 = encoder_block(in_channels, layers_stack[0], conv_layer=Kanvolution2d)
        self.enc2 = encoder_block(layers_stack[0], 2*layers_stack[0], conv_layer=Kanvolution2d)
        self.enc3 = res_encoder_block(layers_stack[1], 2*layers_stack[1], conv_layer=Kanvolution2d)
        self.enc4 = encoder_block(layers_stack[2], 2*layers_stack[2], conv_layer=Kanvolution2d)
        
        # Bottleneck block
        self.bottleneck = ResidualBlock(layers_stack[3], 2*layers_stack[3], conv_layer=Kanvolution2d)
        
        # Decoder blocks
        self.dec1 = decoder_block(2*layers_stack[3], layers_stack[3], conv_layer=Kanvolution2d)
        self.dec2 = res_decoder_block(2*layers_stack[2], layers_stack[2], conv_layer=Kanvolution2d)
        self.dec3 = decoder_block(2*layers_stack[1], layers_stack[1], conv_layer=Kanvolution2d)
        self.dec4 = decoder_block(2*layers_stack[0], layers_stack[0], conv_layer=Kanvolution2d)
        
        # 1x1 convolution
        self.out = nn.Conv2d(layers_stack[0], num_classes, kernel_size=1, padding='same')
        
    def forward(self, image):
        n1, s1 = self.enc1(image)
        n2, s2 = self.enc2(n1)
        n3, s3 = self.enc3(n2)
        n4, s4 = self.enc4(n3)
        
        n5 = self.bottleneck(n4)
        
        n6 = self.dec1(n5, s4)
        n7 = self.dec2(n6, s3)
        n8 = self.dec3(n7, s2)
        n9 = self.dec4(n8, s1)
        
        output = self.out(n9)
        
        return output
    

if __name__ == '__main__':
    divsor = 4
    layers_stack = (16//divsor, 32//divsor, 64//divsor, 128//divsor)
    model = UNet(layers_stack=layers_stack)
    # print(model)

    ### counting params
    print(sum([i.numel() for i in model.parameters() if i.requires_grad]))
    ###

    model = UKonvNet(layers_stack=layers_stack)
    # print(model)
    ### counting params
    print(sum([i.numel() for i in model.parameters() if i.requires_grad]))
    ###