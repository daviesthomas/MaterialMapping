"""
UtilityFunctions.py

Faisal Habib
March 17, 2020

Description:
A set of utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def save_loss_plot(losses, title, filename):
    """
    Saves a plot of the training and validation curves
    :param losses: A dictionary of training losses and validation losses.
    :param title: Title of the graph
    :param filename: Filename to save as (must have the extension .pdf)
    :return: None
    """
    plt.figure()

    for name, data in losses.items():
        plt.plot(range(len(data)), data, label=name)

    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_tensor_image(image_tensor, image_path, aspect_ratio=1.0):
    """
    Saves a tensor image to disk
    :param image_tensor: input tensor that is the image
    :param image_path:  the path of the image
    :param aspect_ratio: default 1.0
    :return: None
    """

    img_numpy = image_tensor.numpy()  # convert it into a numpy array
    img_numpy = img_numpy[0]

    # post-processing: transpose and scaling
    img_numpy = (np.transpose(img_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    img_numpy = img_numpy.astype(np.uint8)

    image_pil = Image.fromarray(img_numpy)
    h, w, _ = img_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    image_pil.save(image_path)
