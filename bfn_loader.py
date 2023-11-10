import math
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import random

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class BFNDataset(Dataset): 
    '''
    your folder should have following structrues:
    
    root
        --1.jpg
        --2.jpg
        --...
    '''
    def __init__(self,
                 root: str,
                 imgSize=[32,32],
                 sigma1=0.001,
                 tmin = 1e-4,
                 labels = '/home/jiayinjun/Bayesian_Flow_Networks/tmp/OxfordFlowersLabels.npy'):

        self.root = root
        self.width, self.height = imgSize
        self.imgList = os.listdir(root)
        self.list = os.listdir(root)
        self.sigma1 = sigma1
        self.labels = np.load(labels)
        self.cat_num = max(self.labels)-min(self.labels)+1 
        self.tmin = tmin
    def __getitem__(self, index,t=None):
        #index = random.randint(0,9)
        filename = self.imgList[int(index)]
        if random.random() > 0: #1/self.cat_num: #mask some label
            label = int(filename.split('.')[0].split('_')[1])
            label = self.labels[label-1]
        else:
            label = 0 #label always from 1
        path = os.path.join(self.root,filename)
        img = pil_loader(path)
        img = F.to_tensor(img)
        img = F.resize(img,(self.width,self.height),antialias=False)
        img = img*2 -1 #[xmin, xmax] = [−1, 1]
        if t is None:
            t = np.random.rand()*(1-self.tmin)+self.tmin
        gamma = 1-self.sigma1**(2*t)
        noise = torch.normal(0,math.sqrt(gamma*(1-gamma)),img.shape)
        noisy_img = gamma*img+noise
        return noisy_img,img,t,gamma,label

    def __len__(self):
        return len(self.imgList)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = BFNDataset("/data/protein/OxfordFlowers/train",[64,64],sigma1=0.01)
    for i in range(0,2000,20):
        print(i)
        img,noisy_img,t,gamma,label = dataset.__getitem__(0,i/2000)
        img = (img + 1)/2
        noisy_img = (noisy_img + 1)/2
        img,noisy_img = F.to_pil_image(img),F.to_pil_image(noisy_img)
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax2.imshow(noisy_img)
        plt.savefig('/home/jiayinjun/Bayesian_Flow_Networks/tmp/test_dataloader.png')