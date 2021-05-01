'''
Code of 'Dual-Cross Central Difference Network for Face Anti-Spoofing' 
By Zitong Yu, 2021

If you use the code, please cite:
@inproceedings{yu2021dual,
    title={Dual-Cross Central Difference Network for Face Anti-Spoofing},
    author={Yu, Zitong and Qin, Yunxiao and ZHoa, Hengshuang and Li, Xiaobai and Zhao, Guoying},
    booktitle= {IJCAI},
    year = {2021}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2021 
'''

import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np




class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff



class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], tensor_zeros, self.conv.weight[:,:,:,2], tensor_zeros, self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff




class C_CDN(nn.Module):
    
    def __init__(self, basic_conv=Conv2d_Hori_Veri_Cross, theta=0.8):   
        super(C_CDN, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            #nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),    
        )
        
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	  
        x_Block2_32x32 = self.downsample32x32(x_Block2)   # x [128, 32, 32]  
        
        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]  	
        x_Block3_32x32 = self.downsample32x32(x_Block3)   # x [128, 32, 32]  
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        depth = x.squeeze(1)
        
        return depth





class DC_CDN(nn.Module):

    def __init__(self, basic_conv1=Conv2d_Hori_Veri_Cross, basic_conv2=Conv2d_Diag_Cross, theta=0.8):   
        super(DC_CDN, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv1(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv1(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv1(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv1(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv1(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv1(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv1(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv1(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv1(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv1(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv1(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv1(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            #basic_conv1(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),    
        )
        
        
        
        # 2nd stream
        self.conv1_2 = nn.Sequential(
            basic_conv2(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1_2 = nn.Sequential(
            basic_conv2(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv2(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv2(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
        )
        
        self.Block2_2 = nn.Sequential(
            basic_conv2(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv2(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv2(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.Block3_2 = nn.Sequential(
            basic_conv2(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv2(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv2(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.lastconv1_2 = nn.Sequential(
            basic_conv2(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2_2 = nn.Sequential(
            basic_conv2(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        #self.lastconv3_2 = nn.Sequential(
        #    basic_conv2(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #    #nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.ReLU(),    
        #)
        
        #self.HP_branch1 = Parameter(torch.ones([3,1]))
        self.HP_branch1 = Parameter(torch.zeros([3,1]))
        #self.HP_branch2 = Parameter(torch.ones([3,1]))
        self.HP_branch2 = Parameter(torch.zeros([3,1]))
        
        
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        
        # 1st stream
        x = self.conv1(x_input)	
        x_2 = self.conv1_2(x_input)	   
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_2 = self.Block1_2(x_2)	    	    	# x [128, 128, 128]
        
        # fusion1
        x_Block1_new = F.sigmoid(self.HP_branch1[0])*x_Block1 + (1-F.sigmoid(self.HP_branch1[0]))*x_Block1_2
        x_Block1_2_new = F.sigmoid(self.HP_branch2[0])*x_Block1_2 + (1-F.sigmoid(self.HP_branch2[0]))*x_Block1
        
        
        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	  
        x_Block2_2 = self.Block2_2(x_Block1_2)	    # x [128, 64, 64]	  
        
        # fusion2
        x_Block2_new = F.sigmoid(self.HP_branch1[1])*x_Block2 + (1-F.sigmoid(self.HP_branch1[1]))*x_Block2_2
        x_Block2_2_new = F.sigmoid(self.HP_branch2[1])*x_Block2_2 + (1-F.sigmoid(self.HP_branch2[1]))*x_Block2
        
        
        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]  	
        x_Block3_2 = self.Block3_2(x_Block2_2)	    # x [128, 32, 32]  	
        
        # fusion3
        x_Block3_new = F.sigmoid(self.HP_branch1[2])*x_Block3 + (1-F.sigmoid(self.HP_branch1[2]))*x_Block3_2
        x_Block3_2_new = F.sigmoid(self.HP_branch2[2])*x_Block3_2 + (1-F.sigmoid(self.HP_branch2[2]))*x_Block3
         
        
        x_Block1_32x32 = self.downsample32x32(x_Block1_new)   # x [128, 32, 32]  
        x_Block2_32x32 = self.downsample32x32(x_Block2_new)   # x [128, 32, 32]  
        x_Block3_32x32 = self.downsample32x32(x_Block3_new)   # x [128, 32, 32] 
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
        x = self.lastconv1(x_concat)    # x [128, 32, 32] 
        depth1 = self.lastconv2(x)    # x [64, 32, 32] 
        #x = self.lastconv3(x)    # x [1, 32, 32] 
        
        #map_x_1 = x.squeeze(1)
        
        
        # 2nd stream
        x_Block1_32x32 = self.downsample32x32(x_Block1_2_new)   # x [128, 32, 32]  
        x_Block2_32x32 = self.downsample32x32(x_Block2_2_new)   # x [128, 32, 32]  
        x_Block3_32x32 = self.downsample32x32(x_Block3_2_new)   # x [128, 32, 32]  
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
        x = self.lastconv1_2(x_concat)    # x [128, 32, 32] 
        depth2 = self.lastconv2_2(x)    # x [64, 32, 32] 
        
        
        # fusion
        depth = torch.cat((depth1,depth2), dim=1)
        depth = self.lastconv3(depth)    # x [1, 32, 32] 
        
        depth = depth.squeeze(1)
        
        
        return depth



if __name__ == '__main__':
    
    inputs = torch.randn(1,3,256,256).cuda()
    
    model_C_CDN = C_CDN(basic_conv=Conv2d_Hori_Veri_Cross, theta=0.8).cuda()
    depth =  model_C_CDN(inputs)
    
    model_C_CDN = C_CDN(basic_conv=Conv2d_Diag_Cross, theta=0.8).cuda()
    depth =  model_C_CDN(inputs)
    
    model_DC_CDN = DC_CDN(basic_conv1=Conv2d_Hori_Veri_Cross, basic_conv2=Conv2d_Diag_Cross, theta=0.8).cuda()
    depth =  model_DC_CDN(inputs)
    
    pdb.set_trace()
    
    
    