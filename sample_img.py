import torch
import os,sys
get_path = os.path.dirname(__file__)
sys.path.append(get_path)
current_path = os.path.dirname(__file__).split('/')
import torch
from ViT import BFN_U_Vit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as transF
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here")

    parser.add_argument("--device", type=str, default="cuda:0", help="Device for computation, default is 'cuda:0'")
    parser.add_argument("--load", type=str, default="last", help="Specify the epoch to load, either 'best' or 'last'")
    parser.add_argument("--SavedDir", type=str, help="Directory to save images")
    parser.add_argument("--ExpConfig", type=str, help="Path to the YAML file of your experiments")
    parser.add_argument("--n_sqrt", type=int, default=8, help="N**2, how many samples you will get")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps for sampling, default is 200")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    load = args.load
    SavedDir = args.SavedDir
    ExpDir = args.ExpConfig
    n_sqrt = args.n_sqrt
    steps = args.steps

    with open(ExpDir,'r') as f:
        training_parameters = yaml.full_load(f)
    TrainDir = training_parameters['dataStorage']
    try:
        os.mkdir(SavedDir)
    except:
        pass
    modelName = training_parameters["framework"]
    image_size = training_parameters["image_size"]
    sigma1 = training_parameters["sigma1"] 
    patch_size = training_parameters["patch_size"] 
    embed_dim = training_parameters["embed_dim"] 
    depth = training_parameters["depth"] 
    head = training_parameters["head"]    
    model = BFN_U_Vit(img_size=image_size,patch_size=patch_size,embed_dim=embed_dim,depth=depth,num_heads=head,sigma1=sigma1)

    if load == 'best':
        initializing = os.path.join(os.path.dirname(ExpDir),'bestloss.pkl')
        state = torch.load(initializing,map_location=device)
    elif load == 'last':
        initializing = os.path.join(os.path.dirname(ExpDir),'lastepoch.pkl')
        state = torch.load(initializing,map_location=device)
        state = state['state_dict']
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state,prefix='module.')

    model.load_state_dict(state,strict=True)
    model.to(device)
    model.eval()
    img_size = model.img_size

    for imageclass in range(102):
        img,labels = model.sampler(device,steps,n_sqrt**2,imageclass+1)
        fig = plt.figure(figsize=img_size)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(n_sqrt, n_sqrt),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        for ax, im,l in zip(grid,img,labels):
            # Iterating over the grid returns the Axes.
            ax.imshow(transF.to_pil_image(im))
            #ax.text(0,8,str(l),fontsize=60, color='blue')
        plt.savefig(os.path.join(SavedDir,f"samples_{imageclass+1}.png"),bbox_inches='tight')
        plt.close()


