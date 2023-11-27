# Denoising_Diffusion_Implicit_Models
A replication of Denoising Diffusion Implicit Models paper with PyTorch and [U-ViT](https://arxiv.org/pdf/2209.12152.pdf).

![img](https://github.com/EBGU/Denoising_Diffusion_Implicit_Models/blob/main/Saved_Models/samples_flower.png)

![img](https://github.com/EBGU/Denoising_Diffusion_Implicit_Models/blob/main/Saved_Models/denoise_sequence_flower.png)

To train a new model, you can modify the yaml file and:

` python multi_gpu_trainer.py example `

Or you can download my pretrained weights for [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/) and run inference:

` python sample_img.py `

The inference process is controled by 6 parameters :

"device", usually `torch.device('cuda')` ;

"load", best epoch or last epoch;

"SavedDir", where to save images;

"ExpDir", the yaml file of your experiments;

"n_sqrt", N**2 how many samples you will get;

"steps", n steps for sampling, the orignal paper recommands 25, but in my experiment, 200 is better.

The result should looks like the welcoming images.

Enjoy!

