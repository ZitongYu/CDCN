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
import imgaug.augmenters as iaa


 




# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])





# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, image_ir, image_depth, binary_mask, spoofing_label = sample['image_x'], sample['image_ir'], sample['image_depth'], sample['binary_mask'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
                    image_ir[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    image_ir[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    image_ir[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
                    image_depth[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    image_depth[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    image_depth[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img,'image_ir': image_ir,'image_depth': image_depth, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, image_ir, image_depth, binary_mask, spoofing_label = sample['image_x'], sample['image_ir'], sample['image_depth'], sample['binary_mask'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        image_ir *= mask
        image_depth *= mask
        return {'image_x': img,'image_ir': image_ir,'image_depth': image_depth, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, spoofing_label = sample['image_x'], sample['image_ir'], sample['image_depth'], sample['binary_mask'],sample['spoofing_label']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_image_ir = (image_ir - 127.5)/128     # [-1,1]
        new_image_depth = (image_depth - 127.5)/128     # [-1,1]

        return {'image_x': new_image_x,'image_ir': new_image_ir,'image_depth': new_image_depth, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, spoofing_label = sample['image_x'], sample['image_ir'], sample['image_depth'], sample['binary_mask'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_image_ir = np.zeros((256, 256, 3))
        new_image_depth = np.zeros((256, 256, 3))
        new_binary_mask = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_image_ir = cv2.flip(image_ir, 1)
            new_image_depth = cv2.flip(image_depth, 1)
            new_binary_mask = cv2.flip(binary_mask, 1)
           
                
            return {'image_x': new_image_x,'image_ir': new_image_ir,'image_depth': new_image_depth, 'binary_mask': new_binary_mask, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x,'image_ir': image_ir,'image_depth': image_depth, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, spoofing_label = sample['image_x'], sample['image_ir'], sample['image_depth'], sample['binary_mask'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        image_ir = image_ir[:,:,::-1].transpose((2, 0, 1))
        image_ir = np.array(image_ir)
        
        image_depth = image_depth[:,:,::-1].transpose((2, 0, 1))
        image_depth = np.array(image_depth)
        
        
        
        binary_mask = np.array(binary_mask)

                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'image_ir': torch.from_numpy(image_ir.astype(np.float)).float(), 'image_depth': torch.from_numpy(image_depth.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.float)).float()}



class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
        
        videoname_ir = videoname[:18] + 'ir/' + videoname[-8:]
        ir_path = os.path.join(self.root_dir, videoname_ir)
        
        videoname_depth = videoname[:18] + 'depth/' + videoname[-8:]
        depth_path = os.path.join(self.root_dir, videoname_depth)
    
    
             
        image_x, image_ir, image_depth, binary_mask = self.get_single_image_x(image_path, ir_path, depth_path)
        
        
		    
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros((32, 32))    
        
        
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  

        sample = {'image_x': image_x, 'image_ir': image_ir, 'image_depth': image_depth, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, ir_path, depth_path):
        
        
        image_x = np.zeros((256, 256, 3))
        binary_mask = np.zeros((32, 32))
 
 
        image_x_temp = cv2.imread(image_path)
        image_x_temp_ir = cv2.imread(ir_path)
        image_x_temp_depth = cv2.imread(depth_path)
        image_x_temp_gray = cv2.imread(image_path, 0)


        image_x = cv2.resize(image_x_temp, (256, 256))
        image_x_ir = cv2.resize(image_x_temp_ir, (256, 256))
        image_x_depth = cv2.resize(image_x_temp_depth, (256, 256))
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        image_x_aug = seq.augment_image(image_x) 
        image_x_aug_ir = seq.augment_image(image_x_ir) 
        image_x_aug_depth = seq.augment_image(image_x_depth) 
        
             
        
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        
        
        
   
        return image_x_aug, image_x_aug_ir, image_x_aug_depth, binary_mask




