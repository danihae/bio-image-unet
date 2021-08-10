import numpy as np
import tifffile


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
    tifffile.imsave(filename, imgs)
    print('Saving prediction results as %s' % filename)
