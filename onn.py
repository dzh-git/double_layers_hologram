import torch
import numpy as np
import parameters
import random
import torch.nn as nn
import utils


class DiffractiveLayer(torch.nn.Module):
    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        args=parameters.my_parameters().get_hyperparameter()
        actual_situation = parameters.my_parameters().get_actualparameter()
        distance=actual_situation.distance
        wave_length=actual_situation.wave_length
        screen_length=actual_situation.screen_length
        wave_num=2*3.14159/wave_length

        #dx表示衍射层像素大小，1/2dx为最大空间采样频率，不改这个
        point_num=args.img_size; dx=screen_length/point_num

        fx_list=np.arange(-1/(2*dx),1/(2*dx),1/screen_length)
        phi = np.fromfunction(
            lambda i, j: 1-(np.square(wave_length*fx_list[i])+np.square(wave_length*fx_list[j])),
            shape=(point_num, point_num), dtype=np.int16).astype(np.complex64)
        H = np.exp(1.0j * wave_num * distance*np.sqrt(phi))
        self.H=torch.fft.fftshift(torch.complex(torch.from_numpy(H.real),torch.from_numpy(H.imag)), dim=(0,1))
        self.H = torch.nn.Parameter( self.H, requires_grad=False)
        
    #在频域进行计算，看信息光学
    def forward(self, waves):
        temp = torch.fft.fft2(torch.fft.fftshift(waves, dim=(1,2)), dim=(1,2))
        k_space=temp.mul(self.H)
        x_space = torch.fft.ifftshift(torch.fft.ifft2(k_space, dim=(1,2)), dim=(1,2))
        return x_space

class DiffraFouriourLayer(torch.nn.Module):
    def __init__(self):
        super(DiffraFouriourLayer, self).__init__()
        
    #在频域进行计算，看信息光学
    def forward(self, waves):
        k_space=torch.fft.fft2(torch.fft.fftshift(waves, dim=(1,2)), dim=(1,2))
        x_space = torch.fft.ifftshift(k_space, dim=(1,2))
        return x_space


class TransmissionLayer(torch.nn.Module):
    def __init__(self):
        super(TransmissionLayer, self).__init__()
        self.args=parameters.my_parameters().get_hyperparameter()
        self.actual_situation = parameters.my_parameters().get_actualparameter()
        t_ones=torch.ones(size=[self.args.img_size,self.args.img_size]).float()
        #表示对L偏振引入的相位
        self.alpha1 = torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)
        self.alpha2 = torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)
        #delta1=Pi
        self.delta2=torch.nn.Parameter(torch.from_numpy(2*np.pi * np.random.random(size=[self.args.img_size,self.args.img_size]).astype('float32')),requires_grad=True)        
        # self.delta=torch.nn.Parameter(np.pi * t_ones,requires_grad=True)
        self.t_zeros=torch.nn.Parameter(torch.zeros(size=[self.args.img_size,self.args.img_size]).float(),requires_grad=False)
        self.matrixI=torch.nn.Parameter(torch.complex(self.t_zeros, t_ones),requires_grad=False)

        self.dropout = nn.Dropout(p=0.1, inplace=False)
        #左右旋
        self.grometry_mask=torch.nn.Parameter(
            torch.stack([torch.ones([self.args.img_size,self.args.img_size]),-torch.ones([self.args.img_size,self.args.img_size])],0)
            .float(),requires_grad=False)
        

    def forward(self, x):
        if self.actual_situation.manufacturing_error:
            mask =self.phase + torch.from_numpy(np.random.random(size=[self.args.img_size
                    ,self.args.img_size]).astype('float32')).cuda()*random.choice([1,-1])*2
        delta2=utils.dequantize_tensor(utils.quantize_tensor(self.delta2))
        cos_delta2=torch.cos(delta2/2)
        sin_delta2=torch.sin(delta2/2)
        cos_alpha1=torch.cos(self.alpha1)
        sin_alpha1=torch.sin(self.alpha1)
        cos_sub=torch.cos(self.alpha1-self.alpha2)
        sin_sub=torch.sin(self.alpha1-self.alpha2)
        #左旋圆偏光入射，求输出
        left_real=torch.zeros_like(x)
        left_imag=torch.zeros_like(x)
        left_real[0,:,:]=-cos_sub*sin_delta2
        left_real[1,:,:]=-sin_sub*sin_delta2
        left_imag[0,:,:]=-cos_alpha1*cos_delta2
        left_imag[1,:,:]=sin_alpha1*cos_delta2
        left_output=torch.complex(left_real,left_imag)
        #右旋圆偏光入射，求输出
        right_real=torch.zeros_like(x)
        right_imag=torch.zeros_like(x)
        right_real[0,:,:]=sin_sub*sin_delta2
        right_real[1,:,:]=-cos_sub*sin_delta2
        right_imag[0,:,:]=sin_alpha1*cos_delta2
        right_imag[1,:,:]=cos_alpha1*cos_delta2
        right_output=torch.complex(right_real,right_imag)
        return (left_output,right_output)

class DTLayer(torch.nn.Module):
    def __init__(self):
        super(DTLayer,self).__init__()
        self.dif=DiffractiveLayer()
        self.tra=TransmissionLayer()

    def forward(self,x):
        x=self.dif(x)
        x=self.tra(x)
        return x

class Net(torch.nn.Module):
    """
    phase only modulation
    """
    def __init__(self, num_layers=1):
        super(Net, self).__init__()
        self.tra=TransmissionLayer()
        self.dif=DiffractiveLayer()
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.actual_situation = parameters.my_parameters().get_actualparameter()
        self.args=parameters.my_parameters().get_hyperparameter()
        
    def forward(self, x):
        # x (200, 200)  torch.complex64
        #表示斜入射
        if self.actual_situation.oblique_incidence:
            #光源随机角度+-4°
            random_thetax=np.random.random()*random.choice([1,-1])*2
            random_thetay=np.random.random()*random.choice([1,-1])*2
            wave_length=self.actual_situation.wave_length
            screen_length=self.actual_situation.screen_length
            wave_num=2*3.14159/wave_length
            dx=screen_length/self.args.img_size
            x_list=np.arange(-0.5*screen_length,0.5*screen_length,dx)

            tilt_phase = torch.from_numpy(np.fromfunction(
            lambda i, j: wave_num*(x_list[i]*np.sin(random_thetax)+x_list[j]*np.sin(random_thetay)),
            shape=(self.args.img_size, self.args.img_size), dtype=np.int16).astype(np.float32))
            mask=torch.complex(torch.cos(tilt_phase), torch.sin(tilt_phase)).cuda()
            x=torch.mul(x,mask)
        left_x,right_x=self.tra(x)
        left_x=self.dif(left_x)
        right_x=self.dif(right_x)
        return (left_x,right_x)
        
        res_angle=torch.angle(x[1,:,:])-torch.angle(x[0,:,:])   #右旋-左旋
        res_angle=res_angle%(2*torch.pi)

        x_abs=abs(x)
        return (x_abs,res_angle)

if __name__=="__main__":
    # x=torch.randn((2,1,51,51))
    # detect_region(x)
    pp=parameters.my_parameters().get_hyperparameter()
    print(pp.img_size)

