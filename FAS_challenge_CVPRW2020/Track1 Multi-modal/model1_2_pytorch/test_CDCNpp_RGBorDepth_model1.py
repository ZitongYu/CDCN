from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from models.CDCNs import Conv2d_cd, CDCNpp

from Loadtemporal_valtest_3modality import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances



# Dataset root     
image_dir = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/'  
   

test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@1_test_res.txt'

#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_test_res.txt'
#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_test_res.txt'


# main function
def train_test():


    print("test:\n ")

     
    #model = CDCNpp( basic_conv=Conv2d_cd, theta=0.7)
	model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
    
    model.load_state_dict(torch.load('CDCNpp_BinaryMask_P1_07/CDCNpp_BinaryMask_P1_07_50.pkl'))


    model = model.cuda()

    print(model) 
    


    model.eval()
    
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
        
        map_score_list = []
        
        for i, sample_batched in enumerate(dataloader_val):
            
            print(i)
            
            inputs = sample_batched['image_x'].cuda()
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
            
            
            map_score = 0.0
            for frame_t in range(inputs.shape[1]):
				if args.modal ==1:
					map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
				if args.modal ==2:
					map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs_depth[:,frame_t,:,:,:])
                score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                map_score += score_norm
            map_score = map_score/inputs.shape[1]
            
            if map_score>1:
                map_score = 1.0

            map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
            
        map_score_val_filename = args.log+'/'+ args.log+ '_map_score_test_50.txt'
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)                
                

    print('Finished testing')
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_BinaryMask_P1_07", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
	parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
	parser.add_argument('--modal', type=int, default=1, help='1: RGB, 2: Depth')

    args = parser.parse_args()
    train_test()
