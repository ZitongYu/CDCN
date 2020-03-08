from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 


frames_total = 8    # each video 8 uniform samples


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, string_name = sample['image_x'], sample['image_ir'], sample['image_depth'],sample['binary_mask'],sample['string_name']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_image_ir = (image_ir - 127.5)/128     # [-1,1]
        new_image_depth = (image_depth - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x,'image_ir': new_image_ir,'image_depth': new_image_depth, 'binary_mask': binary_mask, 'string_name': string_name}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, string_name = sample['image_x'], sample['image_ir'], sample['image_depth'],sample['binary_mask'],sample['string_name']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        
        image_ir = image_ir[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_ir = np.array(image_ir)
        
        image_depth = image_depth[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_depth = np.array(image_depth)
                        
        binary_mask = np.array(binary_mask)
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'image_ir': torch.from_numpy(image_ir.astype(np.float)).float(), 'image_depth': torch.from_numpy(image_depth.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'string_name': string_name} 



class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):

        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
        image_path2 = os.path.join(image_path, 'profile')
        ir_path = os.path.join(image_path, 'ir')
        depth_path = os.path.join(image_path, 'depth')
             
        image_x, image_ir, image_depth, binary_mask = self.get_single_image_x(image_path2, ir_path, depth_path, videoname)
		    
            
        sample = {'image_x': image_x,'image_ir': image_ir,'image_depth': image_depth, 'binary_mask': binary_mask, 'string_name': videoname}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, ir_path, depth_path, videoname):

        files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        interval = files_total//frames_total
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        image_ir = np.zeros((frames_total, 256, 256, 3))
        image_depth = np.zeros((frames_total, 256, 256, 3))
        
        binary_mask = np.zeros((frames_total, 32, 32))
        
        
        
        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            
            s = "%04d.jpg" % image_id            
            
            # RGB
            image_path2 = os.path.join(image_path, s)
            image_x_temp = cv2.imread(image_path2)
            
            # ir
            image_path2_ir = os.path.join(ir_path, s)
            image_x_temp_ir = cv2.imread(image_path2_ir)
            
            # depth
            image_path2_depth = os.path.join(depth_path, s)
            image_x_temp_depth = cv2.imread(image_path2_depth)
            
            image_x_temp_gray = cv2.imread(image_path2, 0)
            image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))

            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256))
            image_ir[ii,:,:,:] = cv2.resize(image_x_temp_ir, (256, 256))
            image_depth[ii,:,:,:] = cv2.resize(image_x_temp_depth, (256, 256))
            
            #print(image_path2)
            
            for i in range(32):
                for j in range(32):
                    if image_x_temp_gray[i,j]>0:
                        binary_mask[ii, i, j]=1.0
                    else:
                        binary_mask[ii, i, j]=0.0
            

        
        return image_x, image_ir, image_depth, binary_mask







if __name__ == '__main__':
    # usage
    # MAHNOB
    root_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/cropped_frm/'
    trainval_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/ImageSet_5fold/trainval_zitong_fold1.txt'
    

    BioVid_train = BioVid(trainval_list, root_list, transform=transforms.Compose([Normaliztion(), Rescale((133,108)),RandomCrop((125,100)),RandomHorizontalFlip(),  ToTensor()]))
    
    dataloader = DataLoader(BioVid_train, batch_size=1, shuffle=True, num_workers=8)
    
    # print first batch for evaluation
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_x'].size(), sample_batched['video_label'].size())
        print(i_batch, sample_batched['image_x'], sample_batched['pain_label'], sample_batched['ecg'])
        pdb.set_trace()
        break

            
 


