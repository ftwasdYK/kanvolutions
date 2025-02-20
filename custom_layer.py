from torch import nn
import torch
from torch.distributions.uniform import Uniform
import numpy as np
from torch.nn import functional as F

class Kanvolution2d(nn.Module):
    """
    Kanvolution2d is a convolutional layer that uses a φ(x) kernel to convolve over the input tensor.
    The kernel is defined by the order of the polynomial and the weights of the kernel are initialized based on the order of the polynomial.
    """
    
    def __init__(
            self, 
            in_channels:int, out_channels:int, 
            order_p:int=3, order_q:int=2, 
            kernel_size: int|tuple= (3,3), 
            stride:int=1, 
            padding:int=0, 
            dilation:int=1, 
            scale:bool= False, 
            bias:bool=True
        ):
        super(Kanvolution2d, self).__init__()
        
        # init params
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.k_height, self.k_width = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.out_channels = out_channels
        self.in_channels = in_channels
        
        if bias:
            self.order_p = order_p +1
        self.order_q = order_q
        self.scale = scale
        
        # init weights
        w_p, w_q, w_outer = self._init_kernel_weights()
        self.w_p = nn.Parameter(w_p, requires_grad=True)
        self.w_q = nn.Parameter(w_q, requires_grad=True)
        self.w_outer = nn.Parameter(w_outer, requires_grad=True)

        # for the forward pass
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.unfold_w = nn.Unfold(kernel_size=kernel_size, padding=0, stride=1)
    def forward(self, x):
        # get the size of the input tensor
        batch_size, _, height, width = x.size()
        # calculate the size of the output tensor
        h_out, w_out = self._calc_size_after_conv((height, width))

        # start the forward pass
        unfo_x = self.unfold(x)
        out_kernel = []
        # can be done in parallel
        for out_chan in range(self.out_channels):
            summ_p = 0
            # calculation of the numerator from φ(x) for each pixel
            for p in range(self.order_p):
                unfo_w_p = self.unfold_w(self.w_p[out_chan,p,...])
                summ_p += (unfo_w_p * unfo_x**p)

            summ_q = 0
            # calculation of the denominator from φ(x) for each pixel
            for q in range(0, self.order_q):
                unfo_w_q = self.unfold_w(self.w_q[out_chan,q,...])
                summ_q += (unfo_w_q * unfo_x**(q+1))
            
            # calculate the φ(x) for each pixel 
            tl_sum = (summ_p / (1 + summ_q.abs()))
            unfo_w_outer = self.unfold_w(self.w_outer[out_chan,...])
            # w .* φ(x)
            summ_outer = tl_sum * unfo_w_outer
            out_kernel.append(summ_outer.sum(axis=1).view(batch_size, 1, h_out, w_out))

        return torch.cat(out_kernel, dim=1)
        
    
    def _init_kernel_weights(self):
        """
        Initialize the weights of the kernel for the Konvolution layer based on the order of the polynomial combined with pytorch's documentation.
        """

        k = 1 / (self.in_channels * self.k_height * self.k_width)
        # draw uniform from -sqrt(k) to sqrt(k) 
        
        uni = Uniform(-k**(1/2), k**(1/2))
        w_outer = uni.sample((self.out_channels, self.in_channels, *self.kernel_size))

        if self.scale:
            w_p = torch.zeros((self.out_channels, self.order_p, self.in_channels, *self.kernel_size))
            w_q = torch.zeros((self.out_channels, self.order_q, self.in_channels, *self.kernel_size))

            sample_points_per_order = (self.out_channels, self.in_channels,  *self.kernel_size)
            for p in range(self.order_p):
                if p == 0 or p == 1:
                    w_p[:,p,...] = uni.sample(sample_points_per_order)
                    continue  
                w_p[:,p,...] = Uniform(-k**(-1/p), k**(-1/p)).sample(sample_points_per_order)

            for q in range(self.order_q):
                if q == 0:
                    w_q[:,q,...] = uni.sample(sample_points_per_order)
                    continue
                w_q[:,q,...] = Uniform(-k**(-1/(q+1)), k**(-1/(q+1))).sample(sample_points_per_order)

                w_outer = w_outer ** (1/2)
            return w_p, w_q, w_outer

        w_p = uni.sample((self.out_channels, self.order_p, self.in_channels, *self.kernel_size))
        w_q = uni.sample((self.out_channels, self.order_q, self.in_channels, *self.kernel_size))  

        return w_p, w_q, w_outer

    def _calc_size_after_conv(self, shape):
        out_height = int(np.floor( ((shape[0] + 2*self.padding -self.dilation*(self.k_height-1) -1 )/ self.stride) + 1))
        out_width = int(np.floor( ((shape[1] + 2*self.padding -self.dilation*(self.k_width-1) -1 )/ self.stride) + 1))
        return out_height, out_width
