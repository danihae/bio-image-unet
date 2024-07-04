import numpy as np
import tifffile
from PIL import Image
from torch import nn
import torch


def save_as_tif(imgs, filename, normalize=False):
    """
    Save numpy array as tif file

    Parameters
    ----------
    imgs : np.array
        Data array
    filename : str
        Filepath to save result
    normalize : bool
        If true, data is normalized [0, 255]
    """
    if normalize:
        imgs = imgs.astype('float32')
        imgs = imgs - np.nanmin(imgs)
        imgs /= np.nanmax(imgs)
        imgs *= 255
    imgs = imgs.astype('uint8')
    tifffile.imwrite(filename, imgs)


def png_to_grayscale_tiff(png_filename, tiff_filename):
    """
    Read a PNG file, convert it to grayscale, normalize the image data, and save it as a TIFF file.

    Parameters
    ----------
    png_filename : str
        The filename of the PNG file to read.
    tiff_filename : str
        The filename of the TIFF file to save.

    Returns
    -------
    None
    """
    # Read the PNG file
    image = Image.open(png_filename)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Normalize the image data by its maximum value
    normalized_image = np.array(grayscale_image) / np.max(grayscale_image) * 255

    # Convert the normalized image to np.uint8 data type
    normalized_image = normalized_image.astype(np.uint8)

    # Save the normalized image as a TIFF file
    tifffile.imwrite(tiff_filename, normalized_image)


def get_device(print_device=False):
    """
    Determines the most suitable device (CUDA, MPS, or CPU) for PyTorch operations.

    Returns:
    - A torch.device object representing the selected device.
    """
    if torch.backends.cuda.is_built():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():  # only for Apple M1/M2/...
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Warning: No CUDA or MPS device found. Calculations will run on the CPU, "
              "which might be slower.")
    if print_device:
        print(f"Using device: {device}")
    return device


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')  # He initialization
