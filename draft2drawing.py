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
from ViT import BFNVisionTransformer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as transF
import yaml
from PIL import Image  
import time
def img2tensor(path,w,h):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')  
    img = transF.to_tensor(img)
    img = transF.resize(img,(w,h),antialias=False)
    img = img*2 -1        
    return img.unsqueeze_(0)

if __name__ == "__main__":
    device = torch.device('cuda:0')
    # model = DiffusionVisionTransformer(img_size=[64,64],total_steps=2000,patch_size=8,embed_dim=384,depth=7,num_heads=12) #miniImageNet
    # state = torch.load(get_path+"/Saved_Models/miniImageNet.pkl")

    # model = DiffusionVisionTransformer(img_size=[64,64],total_steps=2000,patch_size=4,embed_dim=256,depth=6,num_heads=4) #oxford flower
    # state = torch.load(get_path+"/Saved_Models/OxfordFlower.pkl")

    ExpDir = '/home/jiayinjun/Bayesian_Flow_Networks/Saved_Models/20231108_Lvit_tiny_diffusion/20231108_L.yaml'
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

    model = BFNVisionTransformer(img_size=image_size,patch_size=patch_size,embed_dim=embed_dim,depth=depth,num_heads=head,sigma1=sigma1)
    state = torch.load(initializing)

    model.load_state_dict(state,strict=True)
    model.to(device)
    model.eval()
    img_size = model.img_size
 
    k = 1000
    pred = img2tensor('/home/jiayinjun/Bayesian_Flow_Networks/tmp/draft.png',*image_size).to(device)
    for i in range(1,k):
        t = 0.03
        gamma = 1-sigma1**(2*t)
        mu = torch.normal(0,gamma*(1-gamma),(1,3,img_size[0],img_size[1]),device=device)+gamma*pred
        pred = model.forward(mu,torch.tensor(t,device=device).unsqueeze_(0),torch.tensor(gamma,device=device).unsqueeze_(0))
        #debug
        in_ = (mu.clip(-1,1).cpu()[0]+1)/2
        in_ = transF.to_pil_image(in_)
        out = (pred.clip(-1,1).cpu()[0]+1)/2
        out = transF.to_pil_image(out)
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(in_)
        ax2.imshow(out)
        plt.savefig('/home/jiayinjun/Bayesian_Flow_Networks/tmp/drawing.png')



    # n_sqrt = 8
    # steps = 100
    # img = model.sampler(device,steps,n_sqrt**2)
    # fig = plt.figure(figsize=img_size)
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                 nrows_ncols=(n_sqrt, n_sqrt),  # creates 2x2 grid of axes
    #                 axes_pad=0.1,  # pad between axes in inch.
    #                 )
    # for ax, im in zip(grid,img):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(transF.to_pil_image(im))
    # plt.savefig(os.path.join(SavedDir,"samples.png"),bbox_inches='tight')

    # #show denoise sequence
    # N = 6
    # img = model.diffusion_sequence(device,100,N=N)
    # fig = plt.figure(figsize=(len(img),1.5*N),dpi=300)
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                 nrows_ncols=(N, len(img)),  # creates 2x2 grid of axes
    #                 axes_pad=0.1,  # pad between axes in inch.
    #                 )
    # img = torch.stack(img,dim=0).transpose(0,1).flatten(0,1)
    # for ax, im in zip(grid,img):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(transF.to_pil_image(im))
    # plt.savefig(get_path+"/Saved_Models/denoise_sequence.png",bbox_inches='tight')
