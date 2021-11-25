# Bio Image U-Net

Implementations of U-Net and Siam U-Net for biological image segmentation

### Authors
[Daniel Härtter](daniel.haertter@duke.edu) (Duke University, University of Göttingen) \
[Yuxi Long](longyuxi@live.com) (Duke University) \
Andreas Primeßnig

### Installation
Install from [PyPI](https://pypi.org/project/bio-image-unet/): `pip install bio-image-unet`

### Usage example
[iPython Notebook for getting started with U-Net](https://github.com/danihae/bio-image-unet/blob/master/using_unet.ipynb) \
[iPython Notebook for getting started with Siam U-Net](https://github.com/danihae/bio-image-unet/blob/master/using_siam_unet.ipynb)

### Documentation

TBD

## TODO

### Before publication time
Download some open dataset for tissue segmentation on the Internet and include it with our package. 

### Testing plan 1
Generate test dataset.
One image from each project.
Run test dataset.

### Testing plan 2
Just run model for all the images in their respective dataset.

### Ideas for fixing the translational variance 

Adapted from [https://arxiv.org/abs/1805.12219]

1. The source of translational variance is in either zero padding or non-unary stride max pooling. We can try to remove either one of those.

2. A crazy modification to the network at inference time. Change the input size at inference time. Since the network is fully convolutional, this same network will work. 