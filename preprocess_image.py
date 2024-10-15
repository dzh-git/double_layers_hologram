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
import utils

def load_img(path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img0=255-img0
    _,img0=cv2.threshold(img0, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), dtype=np.uint8)
    img0 = cv2.dilate(img0, kernel, 1)
    img0=cv2.resize(img0,dsize=(1000,1000))/255
    return torch.from_numpy(img0)


def preprocess_L():
    args=parameters.my_parameters().get_hyperparameter()
    path0='./dataset/SuoHui.png'
    origin_img=load_img(path0)
    AL=np.zeros((args.img_size,args.img_size))
    AR=np.zeros((args.img_size,args.img_size))
    delta_phi=  np.zeros((args.img_size,args.img_size))

    one_s2=1/np.sqrt(2)
    pi=torch.pi
    sca_list    =[1,0,one_s2,one_s2,one_s2,one_s2, 0.9   ,0.1   ,0.8    ,0.2]
    del_phi_list=[0,0,0     ,pi    ,0.5*pi,1.5*pi, 1.2*pi,0.3*pi,1.8*pi ,0.8*pi]
    step=args.img_size//10
    for i in range(10):
        sca1=sca_list[i]
        AL[step*i : step*(i+1),:]=origin_img[step*i : step*(i+1),:]*sca1
        AR[step*i : step*(i+1),:]=origin_img[step*i : step*(i+1),:]*np.sqrt(1-sca1*sca1)
        delta_phi[step*i : step*(i+1),:]=del_phi_list[i]
    
    
    # cv2.namedWindow('AL',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('AR',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('delta_phi',cv2.WINDOW_NORMAL)
    # cv2.imshow('AL',AL)
    # cv2.imshow('AR',AR)
    # cv2.imshow('delta_phi',delta_phi/(2*pi))
    # cv2.waitKey()

    np.save('./dataset/AL.npy',AL)
    np.save('./dataset/AR.npy',AR)
    np.save('./dataset/delta_phi.npy',delta_phi)
    np.save('./dataset/amplitude_mask.npy',origin_img)

def preprocess_R():
    args=parameters.my_parameters().get_hyperparameter()
    path0='./dataset/yuanhui.png'
    img0=cv2.imread(path0,cv2.IMREAD_GRAYSCALE)
    img0=255-img0
    _,img0=cv2.threshold(img0, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), dtype=np.uint8)
    img0 = cv2.dilate(img0, kernel, 1)
    origin_img=cv2.resize(img0,dsize=(1000,1000))/255
    # cv2.namedWindow('origin',cv2.WINDOW_NORMAL)
    # cv2.imshow('origin',img0)
    # cv2.waitKey()

    AL=np.zeros((args.img_size,args.img_size))
    AR=np.zeros((args.img_size,args.img_size))
    delta_phi=  np.zeros((args.img_size,args.img_size))

    one_s2=1/np.sqrt(2)
    pi=torch.pi
    sca_list    =[1,0,one_s2,one_s2,one_s2,one_s2, 0.866   ,0   ,0.707    ,0.5]
    del_phi_list=[0,0,0     ,pi    ,0.5*pi,1.5*pi, 1.2*pi,0.3*pi,1.8*pi ,0.8*pi]
    step=args.img_size//10
    for i in range(10):
        sca1=sca_list[i]
        AL[:,step*i : step*(i+1)]=origin_img[:,step*i : step*(i+1)]*sca1
        AR[:,step*i : step*(i+1)]=origin_img[:,step*i : step*(i+1)]*np.sqrt(1-sca1*sca1)
        delta_phi[:,step*i : step*(i+1)]=del_phi_list[i]
    
    # cv2.namedWindow('AL',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('AR',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('delta_phi',cv2.WINDOW_NORMAL)
    
    # cv2.imshow('AL',AL)
    # cv2.imshow('AR',AR)
    # cv2.imshow('delta_phi',delta_phi/(2*pi))
    # cv2.waitKey()

    np.save('./dataset/right_AL.npy',AL)
    np.save('./dataset/right_AR.npy',AR)
    np.save('./dataset/right_delta_phi.npy',delta_phi)
    np.save('./dataset/right_amplitude_mask.npy',origin_img)
    
def test():
    AL=np.load('./dataset/right_AL.npy')
    AR=np.load('./dataset/right_AR.npy')
    delta_phi=np.load('./dataset/right_delta_phi.npy')
    amplitude_mask=np.load('./dataset/right_amplitude_mask.npy')
    cv2.namedWindow('AL',cv2.WINDOW_NORMAL)
    cv2.namedWindow('AR',cv2.WINDOW_NORMAL)
    cv2.namedWindow('delta_phi',cv2.WINDOW_NORMAL)
    cv2.namedWindow('amplitude_mask',cv2.WINDOW_NORMAL)
    cv2.imshow('amplitude_mask',amplitude_mask)
    cv2.imshow('AL',AL)
    cv2.imshow('AR',AR)
    cv2.imshow('delta_phi',delta_phi/(2*np.pi))
    cv2.waitKey()
    
if __name__=='__main__':
    preprocess_R()
    test()