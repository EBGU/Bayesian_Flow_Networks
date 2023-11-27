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

if __name__ == "__main__":
    device = torch.device('cuda:1')
    load = 'best' # or "last"
    SavedDir='/home/jiayinjun/Bayesian_Flow_Networks/tmp/samples'
    ExpDir = '/home/jiayinjun/Bayesian_Flow_Networks/Saved_Models/20231111_Cuvitc_bfn/20231111_C.yaml'
    n_sqrt = 8
    steps = 200

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
        state = torch.load(initializing)
    elif load == 'last':
        initializing = os.path.join(os.path.dirname(ExpDir),'lastepoch.pkl')
        state = torch.load(initializing)
        state = state['state_dict']
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


