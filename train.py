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


def load_img(args,path):
    img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img0=cv2.resize(img0,dsize=(args.img_size,args.img_size))
    #归一化，总能量为1
    img0=img0/np.sum(img0)
    return img0


'''
输入：tensor[2,W,H]，第一个通道和第二个通道代表右旋和左旋。
输出：tensor[2,W,H]，由于几何相位，输出的第0个通道代表左旋，第1个通道代表右旋
相位差delta_phi表示 右旋-左旋
'''
def main(args):
    #模型保存路径
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    left_I_tar,left_stocks_tar,left_amplitude_mask,left_total_energy,left_pixel_num=utils.load_target('left_',device)
    right_I_tar,right_stocks_tar,right_amplitude_mask,right_total_energy,right_pixel_num=utils.load_target('right_',device)
    

    #线偏光入射
    train_images=torch.ones(size=[2,args.img_size,args.img_size]) if device=='cpu'  else torch.ones(size=[2,args.img_size,args.img_size]).cuda()
    train_images=train_images
    
    model=onn.Net()
    # model.load_state_dict(torch.load(r'./saved_model/best.pth'))
    model.to(device)
    
    criterion = torch.nn.MSELoss(reduction='sum') if device == "cpu" else torch.nn.MSELoss(reduction='sum').cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    early_stopping=EarlyStop()
    for epoch in range(args.num_epochs):
        model.train()
        pre_left,pre_right=model(train_images)

        #left
        left_I_pre,left_stocks_pre=utils.convertLR2stocks(pre_left)
        left_I_pre=left_I_pre /torch.sum(left_I_pre) *left_total_energy #能量归一化
        left_stocks_pre=left_stocks_pre*left_amplitude_mask

        # loss_stocks=criterion(stocks_pre,stocks_tar).float()/pixel_num
        left_loss_I1=criterion(left_I_pre,left_I_tar).float()/1e6  #损失：归一化后，整图光强分布误差
        left_loss_s1=criterion(left_stocks_pre[0,:,:],left_stocks_tar[0,:,:]).float()/left_pixel_num
        left_loss_s2=criterion(left_stocks_pre[1,:,:],left_stocks_tar[1,:,:]).float()/left_pixel_num
        left_loss_s3=criterion(left_stocks_pre[2,:,:],left_stocks_tar[2,:,:]).float()/left_pixel_num

        #right
        right_I_pre,right_stocks_pre=utils.convertLR2stocks(pre_right)
        right_I_pre=right_I_pre /torch.sum(right_I_pre) *right_total_energy #能量归一化
        right_stocks_pre=right_stocks_pre*right_amplitude_mask

        # loss_stocks=criterion(stocks_pre,stocks_tar).float()/pixel_num
        right_loss_I1=criterion(right_I_pre,right_I_tar).float()/1e6  #损失：归一化后，整图光强分布误差
        right_loss_s1=criterion(right_stocks_pre[0,:,:],right_stocks_tar[0,:,:]).float()/right_pixel_num
        right_loss_s2=criterion(right_stocks_pre[1,:,:],right_stocks_tar[1,:,:]).float()/right_pixel_num
        right_loss_s3=criterion(right_stocks_pre[2,:,:],right_stocks_tar[2,:,:]).float()/right_pixel_num

        loss_s1=left_loss_s1+ right_loss_s1;loss_s2=left_loss_s2+ right_loss_s2
        loss_s3=left_loss_s3+ right_loss_s3;loss_I1=left_loss_I1+ right_loss_I1
        total_loss=loss_s1 +loss_s2 +loss_s3+loss_I1

        if torch.isnan(total_loss):
            print('loss is nan , break')
            break
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch%50==0:
            print('loss_I1:{:.9f},loss_s1:{:.9f},loss_s2:{:9f},loss_s3:{:9f}'.format(loss_I1,loss_s1,loss_s2,loss_s3))

        early_stopping(-total_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(model.state_dict(),'./saved_model/last.pth')


if __name__=='__main__':
    args=parameters.my_parameters().get_hyperparameter()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    main(args)
    
