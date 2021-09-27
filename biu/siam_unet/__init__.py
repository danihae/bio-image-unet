try:
    import torch
except ImportError:
    raise ImportError('Torch is not installed. Please install through https://pytorch.org/get-started/locally/')
from .data import DataProcess
from .losses import *
from .predict import Predict
from .siam_unet import Siam_UNet
from .train import Trainer
