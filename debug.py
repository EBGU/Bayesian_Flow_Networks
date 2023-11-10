import math
from turtle import mode
import torch
import torch.nn as nn
import warnings
import os,sys
get_path = os.path.dirname(__file__)
sys.path.append(get_path)
current_path = os.path.dirname(__file__).split('/')
import torch
from ViT import BFNVisionTransformer,BFN_U_Vit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as transF
import yaml
from bfn_loader import BFNDataset
if __name__ == "__main__":
    device = torch.device('cuda:3')
    ExpDir = '/home/jiayinjun/Bayesian_Flow_Networks/Saved_Models/20231110uvit_bfn/20231110.yaml'
    with open(ExpDir,'r') as f:
        training_parameters = yaml.full_load(f)

    TrainDir = training_parameters['dataStorage']
    SavedDir='/home/jiayinjun/Bayesian_Flow_Networks/tmp'
    try:
        os.mkdir(SavedDir)
    except:
        pass
    initializing = os.path.join(os.path.dirname(ExpDir),'bestloss.pkl')
    modelName = training_parameters["framework"]
    image_size = training_parameters["image_size"]
    sigma1 = training_parameters["sigma1"] 
    patch_size = training_parameters["patch_size"] 
    embed_dim = training_parameters["embed_dim"] 
    depth = training_parameters["depth"] 
    head = training_parameters["head"]    

    model = BFN_U_Vit(img_size=image_size,patch_size=patch_size,embed_dim=embed_dim,depth=depth,num_heads=head,sigma1=sigma1)
    state = torch.load(initializing)

    model.load_state_dict(state,strict=True)
    model.to(device)
    model.eval()
    img_size = model.img_size

    dataset = BFNDataset("/data/protein/OxfordFlowers/val",img_size,sigma1=sigma1,labels="/data/protein/OxfordFlowers/OxfordFlowersLabels.npy")
    N = 1000
    for i in range(0,N,1):
        print(i)
        noisy_img,img,t,gamma,label = dataset.__getitem__(10,i/N)
        img_,noisy_img_ = transF.to_pil_image((img + 1)/2),transF.to_pil_image((noisy_img + 1)/2)
        #img = img.unsqeeze_(0).to(device)
        noisy_img = noisy_img.unsqueeze_(0).to(device)
        t = torch.tensor(t).float().unsqueeze_(0).to(device)
        gamma = torch.tensor(gamma).float().unsqueeze_(0).to(device)
        label = torch.tensor(label).long().unsqueeze_(0).to(device)
        denoised = torch.clip(model(noisy_img,t,gamma,label).cpu(),-1,1)
        denoised_ = transF.to_pil_image((denoised.squeeze(0) + 1)/2)
        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(img_)
        ax2.imshow(noisy_img_)
        ax3.imshow(denoised_)
        plt.savefig('/home/jiayinjun/Bayesian_Flow_Networks/tmp/debug.png')
