import os
from functools import lru_cache
import numpy as np
import torch
import parameters
import random
import onn
from EarlyStop import EarlyStop
import cv2
import matplotlib.pyplot as plt

def load_img(path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _,img0=cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY)
    img0=cv2.resize(img0,dsize=(100,300))/255
    return torch.from_numpy(img0)

def convertXY2LR(Exy):
    Sq2=1/np.sqrt(2)
    conMat=torch.complex(torch.tensor([[1,0],[1,0]],dtype=torch.float32),torch.tensor([[0,-1],[0,1]],dtype=torch.float32))*Sq2
    if Exy.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Exy.shape
    Exy=torch.reshape(Exy,shape=(2,-1))
    Elr=torch.matmul(conMat,Exy)
    Elr=torch.reshape(Elr,shape=(2,width,height))
    return Elr

def convertAB2LR(Eab):
    conMat=torch.complex(torch.tensor([[1,1],[1,1]],dtype=torch.float32),torch.tensor([[-1,1],[1,-1]],dtype=torch.float32))*0.5
    if Eab.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Eab.shape
    Eab=torch.reshape(Eab,shape=(2,-1))
    Elr=torch.matmul(conMat,Eab)
    Elr=torch.reshape(Elr,shape=(2,width,height))
    return Elr

#将圆偏转xy坐标系。第0通道：L，第一通道：R
def convertLR2XY(Elr):
    Sq2=1/np.sqrt(2)
    conMat=torch.complex(torch.tensor([[1,1],[0,0]],dtype=torch.float32),torch.tensor([[0,0],[1,-1]],dtype=torch.float32))*Sq2
    if Elr.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Elr.shape
    Elr=torch.reshape(Elr,shape=(C,-1))
    Exy=torch.matmul(conMat,Elr)
    Exy=torch.reshape(Exy,shape=(C,width,height))
    return Exy

def convertLR2AB(Elr):
    conMat=torch.complex(torch.tensor([[1,1],[1,1]],dtype=torch.float32),torch.tensor([[1,-1],[-1,1]],dtype=torch.float32))*0.5
    if Elr.device !=conMat.device:
        conMat=conMat.cuda()
    [C,width,height]=Elr.shape
    Elr=torch.reshape(Elr,shape=(2,-1))
    Eab=torch.matmul(conMat,Elr)
    Eab=torch.reshape(Eab,shape=(2,width,height))
    return Eab

def complex2Afai(E0):
    amp=torch.abs(E0)
    fai=torch.angle(E0[1,:,:])-torch.angle(E0[0,:,:])
    fai=(fai+2*torch.pi)%(2*torch.pi)
    return (amp,fai)

#将左右旋转换为stocks参量，这里规定输入的0通道为L，1通道为R
def convertLR2stocks(E0):
    Exy=convertLR2XY(E0)
    Exy_abs,dfai=complex2Afai(Exy)

    Exy_energy=Exy_abs*Exy_abs

    #千万不要忘记归一化，以及对I0很小的值做处理
    I0=Exy_energy[0,:,:]+Exy_energy[1,:,:]
    I0[I0<1e-6]=1e-6

    S1=Exy_energy[0,:,:]-Exy_energy[1,:,:]
    S2=2*Exy_abs[0,:,:]*Exy_abs[1,:,:]*torch.cos(dfai)
    S3=2*Exy_abs[0,:,:]*Exy_abs[1,:,:]*torch.sin(dfai)
    # print("s1:{},s2:{},s3:{}".format(S1,S2,S3))
    #法2,两种方法等价
    # EEE=Exy[0,:,:]*torch.conj(Exy[1,:,:])
    # S22=2*torch.real(EEE)
    # S32=-2*torch.imag(EEE)
    Stocks=torch.stack((S1,S2,S3),0)/I0
    return (I0,Stocks)

def calc_phaseMask(args,path1,path2):
    img1=cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    img1=cv2.resize(img1,dsize=(args.img_size,args.img_size))/255
    img2=cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
    img2=cv2.resize(img2,dsize=(args.img_size,args.img_size))/255
    phase=np.logical_or(img1,img2)
    pm=torch.from_numpy(phase)
    return pm

#量化
def quantize_tensor(x):
    scale=2*np.pi/8
    q_x=x/scale
    q_x=q_x.round()%8
    return q_x
#反量化
def dequantize_tensor(q_x):
    scale=2*np.pi/8
    return scale*q_x

def load_target(flag,device):
    AL=torch.from_numpy(np.load('./dataset/'+flag+'AL.npy'))
    AR=torch.from_numpy(np.load('./dataset/'+flag+'AR.npy'))
    target_A=torch.stack((AL,AR),0).float()
    delta_phi=torch.from_numpy(np.load('./dataset/'+flag+'delta_phi.npy')).float()
    amplitude_mask=np.load('./dataset/'+flag+'amplitude_mask.npy')
    amplitude_mask=torch.from_numpy(amplitude_mask).float()


    if device !='cpu':
        target_A=target_A.cuda()
        delta_phi=delta_phi.cuda()
        amplitude_mask=amplitude_mask.cuda()
    delta_phi=delta_phi *amplitude_mask
    target_A=target_A   *amplitude_mask
    pixel_num=torch.sum(amplitude_mask)
    #目标的stocks参量
    AL_C=torch.complex(target_A[0,:,:].float(),torch.zeros_like(target_A[0,:,:]).float())
    AR_C=torch.complex(target_A[1,:,:]*torch.cos(delta_phi).float(),target_A[1,:,:]*torch.sin(delta_phi).float())
    
    LR=torch.stack((AL_C,AR_C),0)
    I_tar,stocks_tar =convertLR2stocks(LR)
    I_tar=I_tar*amplitude_mask  
    total_energy=torch.sum(I_tar)
    stocks_tar=stocks_tar*amplitude_mask
    return (I_tar,stocks_tar,amplitude_mask,total_energy,pixel_num)
    