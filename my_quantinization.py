import os
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters
import random
import onn
from EarlyStop import EarlyStop
import matplotlib.pyplot as plt
import utils

def quantize_tensor(x,num_bit=3):
    scale=2*np.pi/8
    q_x=x/scale
    q_x=q_x.round()%8
    # qmin=0;qmax=2.**num_bit-1.
    # q_x.clamp_(qmin,qmax).round_()
    # q_x=q_x.round().byte()
    return q_x

def dequantize_tensor(q_x):
    scale=2*np.pi/8
    return scale*q_x.float()

class QuantLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True):
        super(QuantLinear,self).__init__(in_features,out_features,bias)
        self.quant_flag=False
        self.scale=None
        self.zero_point=None

    def linear_quant(self,quantize_bit=3):
        self.weight.data,self.scale,self.zero_point=quantize_tensor(self.weight.data,num_bit=quantize_bit)
        self.quant_flag=True
    
    def forward(self,x):
        if self.quant_flag==True:
            weight=dequantize_tensor(self.weight,self.scale,self.zero_point)
            return F.linear(x,weight,self.bias)
        else:
            return F.linear(x,self.weight,self.bias)
    

if __name__=='__main__':
    w=torch.rand(3,4)*2*torch.pi
    print(w)
    print(dequantize_tensor(*quantize_tensor(w)))




