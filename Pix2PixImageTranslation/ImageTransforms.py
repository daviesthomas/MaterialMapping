"""
ImageTransforms.py

Faisal Habib
March 18, 2020

Description:
This file contains various image transformation functions.
Sections in this file are adopted from base_dataset.py that is available at https://github.com/phillipi/pix2pix
but have been slightly modified for our task
"""

from enum import Enum
from PIL import Image

import random
import numpy as np
import torchvision.transforms as transforms


class PreprocessOptions(Enum):
    """
    Enum for determining the type of preprocessing that needs to be applied to the images.
    """
    NONE = 0,
    RESIZE_AND_CROP = 1,
    SCALE_WIDTH_AND_CROP = 2


def get_preprocess_params(preprocess_options=PreprocessOptions.NONE, image_size=(256, 256), load_size=286,
                          crop_size=256):
    """
    Determines the random position from where to apply the crop and if the image has to be flipped
    The output of this function is used by get_transform.
    The final image will be a square image if any preprocessing is applied.

    :param preprocess_options: enum that specifies the type of preprocessing to be applied
    :param image_size: a tuple representing the image size (w x h)
    :param load_size: a scalar used for resizing the image (to a larger image)
    :param crop_size: a scalar used for cropping the image
    :return: a dictionary of the crop position and flip option
    """

    w, h = image_size
    new_h = h
    new_w = w
    if preprocess_options == PreprocessOptions.RESIZE_AND_CROP:
        new_h = load_size
        new_w = load_size
    elif preprocess_options == PreprocessOptions.SCALE_WIDTH_AND_CROP:
        new_w = load_size
        new_h = load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_position': (x, y), 'flip': flip}


def get_transform(preprocess_options=PreprocessOptions.NONE, params=None, load_size=286, crop_size=256,
                  num_channels=3, grayscale=False, no_flip=False, method=Image.BICUBIC, convert=True):
    """
    Determines all the transforms that have to be applied to images given the options in the parameter list.
    The final image will be a square image.

    :param preprocess_options:  enum that specifies the type of preprocessing to be applied
    :param params:  the output of get_preprocess_params
    :param load_size:  a scalar used for resizing the image (to a larger image)
    :param crop_size:  a scalar used for cropping the image
    :param num_channels:  a scalar used for normalizing tuple length
    :param grayscale:  a boolean that determines if the image has to be converted to a grayscale
    :param no_flip:  a boolean that determines if the image has to be flipped
    :param method:  a type of transform method to apply to images
    :param convert:  a boolean that converts the images into a tensor
    :return: a list of transforms to apply to an image
    """

    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if preprocess_options == PreprocessOptions.RESIZE_AND_CROP:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_position'], crop_size)))
    elif preprocess_options == PreprocessOptions.SCALE_WIDTH_AND_CROP:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))

    # if preprocess_options == PreprocessOptions.NONE:
    #     transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not no_flip:
        if params is not None:
            if params['flip']:
                transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            mean_stddev = tuple(0.5 for i in range(num_channels))
            transform_list += [transforms.Normalize(mean_stddev, mean_stddev)]

    return transforms.Compose(transform_list)


# ----------------------------------------------------------------------------------------------------------------------
# The rest of the code is identical to base_dataset.py
# ----------------------------------------------------------------------------------------------------------------------
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

# ----------------------------------------------------------------------------------------------------------------------
