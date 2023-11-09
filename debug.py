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
from bfn_loader import BFNDataset
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

    dataset = BFNDataset("/drug/OxfordFlowers/val",img_size,sigma1=sigma1)
    N = 100000
    for i in range(0,N,1):
        print(i)
        noisy_img,img,t,gamma = dataset.__getitem__(0,i/N)
        img_,noisy_img_ = transF.to_pil_image((img + 1)/2),transF.to_pil_image((noisy_img + 1)/2)
        #img = img.unsqeeze_(0).to(device)
        noisy_img = noisy_img.unsqueeze_(0).to(device)
        t = torch.tensor(t).float().unsqueeze_(0).to(device)
        gamma = torch.tensor(gamma).float().unsqueeze_(0).to(device)
        denoised = torch.clip(model(noisy_img,t,gamma).cpu(),-1,1)
        denoised_ = transF.to_pil_image((denoised.squeeze(0) + 1)/2)
        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(img_)
        ax2.imshow(noisy_img_)
        ax3.imshow(denoised_)
        plt.savefig('/home/jiayinjun/Bayesian_Flow_Networks/tmp/debug.png')



    # n_sqrt = 8
    # steps = 25
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

    # # #show denoise sequence
    # # N = 6
    # # img = model.diffusion_sequence(device,100,N=N)
    # # fig = plt.figure(figsize=(len(img),1.5*N),dpi=300)
    # # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    # #                 nrows_ncols=(N, len(img)),  # creates 2x2 grid of axes
    # #                 axes_pad=0.1,  # pad between axes in inch.
    # #                 )
    # # img = torch.stack(img,dim=0).transpose(0,1).flatten(0,1)
    # # for ax, im in zip(grid,img):
    # #     # Iterating over the grid returns the Axes.
    # #     ax.imshow(transF.to_pil_image(im))
    # # plt.savefig(get_path+"/Saved_Models/denoise_sequence.png",bbox_inches='tight')
