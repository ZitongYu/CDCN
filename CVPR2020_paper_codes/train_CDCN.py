'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from models.CDCNs import Conv2d_cd, CDCN, CDCNpp

from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances



# Dataset root
train_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Train_images/'        
val_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Dev_images/'     
test_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Test_images/'   
   
map_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/IJCB_re/OULUtrain_images/'   
val_map_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/IJCB_re/OULUdev_images/' 
test_map_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/IJCB_re/OULUtest_images/' 

train_list = '/wrk/yuzitong/DONOTREMOVE/OULU/OULU_Protocols/Protocol_1/Train.txt'
val_list = '/wrk/yuzitong/DONOTREMOVE/OULU/OULU_Protocols/Protocol_1/Dev.txt'
test_list =  '/wrk/yuzitong/DONOTREMOVE/OULU/OULU_Protocols/Protocol_1/Test.txt'
 



# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap( x, feature1, feature2, feature3, map_x):
    ## initial images 
    feature_first_frame = x[0,:,:,:].cpu()    ## the middle frame 

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_visual.jpg')
    plt.close()


    ## first feature
    feature_first_frame = feature1[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_Block1_visual.jpg')
    plt.close()
    
    ## second feature
    feature_first_frame = feature2[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_Block2_visual.jpg')
    plt.close()
    
    ## third feature
    feature_first_frame = feature3[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_Block3_visual.jpg')
    plt.close()
    
    ## third feature
    heatmap2 = torch.pow(map_x[0,:,:],2)    ## the middle frame 

    heatmap2 = heatmap2.data.cpu().numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap2)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_DepthMap_visual.jpg')
    plt.close()
    






def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss




# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log_P1.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')
        log_file.write('finetune!\n')
        log_file.flush()
            
        model = CDCN()
        #model = model.cuda()
        model = model.to(device[0])
        model = nn.DataParallel(model, device_ids=device, output_device=device[0])
        model.load_state_dict(torch.load('xxx.pkl'))

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()


        
        #model = CDCN(basic_conv=Conv2d_cd, theta=0.7)
	model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        


        model = model.cuda()
        #model = model.to(device[0])
        #model = nn.DataParallel(model, device_ids=device, output_device=device[0])

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    


    #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64 

    ACER_save = 1.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = Spoofing_train(train_list, train_image_dir, map_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, map_label, spoof_label = sample_batched['image_x'].cuda(), sample_batched['map_x'].cuda(), sample_batched['spoofing_label'].cuda() 

            optimizer.zero_grad()
            
            #pdb.set_trace()
            
            # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
           
            
            absolute_loss = criterion_absolute_loss(map_x, map_label)
            contrastive_loss = criterion_contrastive_loss(map_x, map_label)
            
            loss =  absolute_loss + contrastive_loss
            #loss =  absolute_loss 
             
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
                #log_file.write('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg))
                #log_file.flush()
                
            #break
        
        # whole epoch average
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
           
    
                    
        #### validation/test
        if epoch <300:
             epoch_test = 300   
        else:
            epoch_test = 20   
        #epoch_test = 1
        if epoch % epoch_test == epoch_test-1:    # test every 5 epochs  
            model.eval()
            
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_val):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    val_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    map_score = 0.0
                    for frame_t in range(inputs.shape[1]):
                        map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                        
                        score_norm = torch.sum(map_x)/torch.sum(val_maps[:,frame_t,:,:])
                        map_score += score_norm
                    map_score = map_score/inputs.shape[1]
                        
                    map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
                    #pdb.set_trace()
                map_score_val_filename = args.log+'/'+ args.log+'_map_score_val.txt'
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)                
                
                ###########################################
                '''                test             '''
                ##########################################
                # test for ACC
                test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    map_score = 0.0
                    for frame_t in range(inputs.shape[1]):
                        map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                        
                        score_norm = torch.sum(map_x)/torch.sum(test_maps[:,frame_t,:,:])
                        map_score += score_norm
                    map_score = map_score/inputs.shape[1]
                        
                    map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
                
                map_score_test_filename = args.log+'/'+ args.log+'_map_score_test.txt'
                with open(map_score_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                #############################################################     
                #       performance measurement both val and test
                #############################################################     
                val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(map_score_val_filename, map_score_test_filename)
                
                print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (epoch + 1, val_threshold, val_ACC, val_ACER))
                log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (epoch + 1, val_threshold, val_ACC, val_ACER))
              
                print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
                #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
                log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
                #log_file.write('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f \n\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
                log_file.flush()
                
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))


    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=7, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=1400, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_P1", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
