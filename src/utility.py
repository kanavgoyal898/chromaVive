import numpy as np
import skimage as ski

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

def load_image(img_path):
    """
    Loads an image from a file path and ensures it has 3 color channels.

    Parameters:
    ----------
    img_path : str
        Path to the image file.

    Returns:
    -------
    numpy.ndarray
        A NumPy array of the image in shape (height, width, 3). 
        If the image is grayscale, it replicates values across 3 channels.
    """
    out = np.asarray(Image.open(img_path))
    if out.ndim == 2:
        out = np.tile(out[:, :, None], 3)
    return out

def resize_image(img, size=(256, 256), resample=3):
    """
    Resizes an image to the specified dimensions with a chosen resampling filter.

    Parameters:
    ----------
    img : numpy.ndarray
        Input image array.
    size : tuple of int, optional
        Target dimensions as (width, height), default is (256, 256).
    resample : int, optional
        Resampling filter to use for resizing, where:
        - 0 (Image.NEAREST): Nearest-neighbor interpolation; fastest but lowest quality, 
          useful for images with sharp edges or simple shapes.
        - 2 (Image.BILINEAR): Linear interpolation; moderate speed and quality, ideal for general resizing.
        - 3 (Image.BICUBIC): Bicubic interpolation; higher quality and ideal for most resizing needs, 
          especially for preserving details.
        - 1 (Image.LANCZOS): Lanczos filter; highest quality, especially for downscaling, but slowest.

    Returns:
    -------
    numpy.ndarray
        Resized image as a NumPy array with the specified dimensions.
    """
    return np.asarray(Image.fromarray(img).resize(size, resample=resample))

def preprocess_image(img_RGB, size=(256, 256), resample=3):
    """
    Preprocesses an RGB image by resizing and converting to LAB color space.

    Parameters:
    ----------
    img_RGB : numpy.ndarray
        Input RGB image array (H, W, 3).
    size : tuple of int, optional
        Target dimensions for resizing (width, height), default is (256, 256).
    resample : int, optional
        Resampling filter for resizing, default is 3 (Image.BICUBIC).

    Returns:
    -------
    tuple of torch.Tensor
        L channel tensors from the original and resized images, both shaped (1, 1, H, W).
    """
    img_RGV_resized = resize_image(img_RGB, size=size, resample=resample)

    img_LAB = ski.color.rgb2lab(img_RGB)
    img_LAB_resized = ski.color.rgb2lab(img_RGV_resized)

    img_L = img_LAB[:, :, 0]                                            # (H, W)
    img_L_resized = img_LAB_resized[:, :, 0]                            # (H, W)

    img_L = torch.tensor(img_L)[None, None, :, :]                       # (B, C, H, W)
    img_L_resized = torch.tensor(img_L_resized)[None, None, :, :]       # (B, C, H, W)

    return img_L, img_L_resized

def postprocess_image(img_L, img_out_AB, mode='bilinear'):
    """
    Combines the L channel with the AB channels and converts LAB to RGB.

    Parameters:
    ----------
    img_L : torch.Tensor
        Input L channel tensor shaped (1, 1, H, W).
    img_out_AB : torch.Tensor
        Input AB channel tensor shaped (1, 2, H_out, W_out).
    mode : str, optional
        Interpolation mode for resizing, default is 'bilinear'.

    Returns:
    -------
    numpy.ndarray
        RGB image converted from the combined LAB channels.
    """
    H, W = img_L.shape[-2], img_L.shape[-1]
    H_out, W_out = img_out_AB.shape[-2], img_out_AB.shape[-1]

    if (H != H_out) or (W != W_out):
        img_out_AB = F.interpolate(img_out_AB, size=(H, W), mode='bilinear')
    
    img_LAB = torch.cat((img_L, img_out_AB), dim=1)                                     # (B, C, H, W)
    return ski.color.lab2rgb(img_LAB.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))   # (H, W, C)