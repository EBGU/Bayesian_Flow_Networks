# Bayesian_Flow_Networks
A replication of [Bayesian_Flow_Networks](https://arxiv.org/abs/2308.07037) paper with PyTorch and [U-ViT](https://arxiv.org/pdf/2209.12152.pdf).

| ![samples_10](images/samples_10.png) | ![samples_41](images/samples_41.png) | ![samples_49](images/samples_49.png) |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![samples_50](images/samples_50.png) | ![samples_62](images/samples_62.png) | ![samples_66](images/samples_66.png) |
| ![samples_73](images/samples_73.png) | ![samples_83](images/samples_83.png) | ![samples100](images/samples100.png) |

To train a new model, you can modify the yaml file and:

` python multi_gpu_trainer.py example `

Training data of [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/) should be split manually, and you can find the numpy version of their labels in this repo.

To run inference:

` python sample_img.py --device "cuda:0" --load "last" --SavedDir tmp/ --ExpConfig example/example.yaml --n_sqrt 8 --steps 200 `

The inference process is controled by 6 parameters :

"device", usually 'cuda:0' ;

"load", best epoch or last epoch;

"SavedDir", where to save images;

"ExpConfig", the yaml file of your experiments;

"n_sqrt", you will get N<sup>2</sup> samples for each class;

"steps", n steps for sampling, the orignal paper recommands 25, but in my experiment, 200 is better.

The result should looks like the welcoming images.

Enjoy!

