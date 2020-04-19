"""
HelperFunctions.py

Faisal Habib
March 3, 2020

Description:
A set of functions to create pix2pix convolution and deconvolution blocks
"""
from enum import Enum
import torch.nn as nn


class NormType(Enum):
    NONE = 0,
    BATCH_NORM = 1,
    INSTANCE_NORM = 2

class ActivationType(Enum):
    NONE = 0,
    RELU = 1,
    LEAKY_RELU = 2,
    SIGMOID = 3,
    TANH = 4


class BlockType(Enum):
    ENCODER = 1,
    DECODER = 2,


def create_block(block_type, in_channels, out_channels, kernel_size, stride, padding, norm_type, activation_type,
                 dropout, spectral_norm):
    convolution = None
    activation = None

    if block_type == BlockType.ENCODER:
        if spectral_norm:
            convolution = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    elif block_type == BlockType.DECODER:
        if spectral_norm:
            convolution = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            convolution = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    else:
        raise Exception('ImageTranslationModel::Unknown Block Type')

    if activation_type == ActivationType.LEAKY_RELU:
        activation = nn.LeakyReLU(0.2)
    elif activation_type == ActivationType.RELU:
        activation = nn.ReLU()
    elif activation_type == ActivationType.SIGMOID:
        activation = nn.Sigmoid()
    elif activation_type == ActivationType.TANH:
        activation = nn.Tanh()
    elif activation_type == ActivationType.NONE:
        activation = None
    else:
        raise Exception('ImageTranslationModel::Unknown Activation Type')

    if dropout:
        if norm_type == NormType.NONE:
            block = [convolution]
            if activation is not None:
                block.extend([activation, nn.Dropout2d(0.5)])
            else:
                block.extend([nn.Dropout2d(0.5)])

        elif norm_type == NormType.BATCH_NORM:
            block = [convolution, nn.BatchNorm2d(out_channels)]
            if activation is not None:
                block.extend([activation, nn.Dropout2d(0.5)])
            else:
                block.extend([nn.Dropout2d(0.5)])

        elif norm_type == NormType.INSTANCE_NORM:
            block = [convolution, nn.InstanceNorm2d(out_channels)]
            if activation is not None:
                block.extend([activation, nn.Dropout2d(0.5)])
            else:
                block.extend([nn.Dropout2d(0.5)])

        else:
            raise Exception('ImageTranslationModel::Unknown Norm Type')
    else:
        if norm_type == NormType.NONE:
            block = [convolution]
            if activation is not None:
                block.extend([activation])

        elif norm_type == NormType.BATCH_NORM:
            block = [convolution, nn.BatchNorm2d(out_channels)]
            if activation is not None:
                block.extend([activation])

        elif norm_type == NormType.INSTANCE_NORM:
            block = [convolution, nn.InstanceNorm2d(out_channels)]
            if activation is not None:
                block.extend([activation])
        else:
            raise Exception('ImageTranslationModel::Unknown Norm Type')

    return block


def initialize_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        # w ~ N(0, 1)*sqrt(1/(fan_in))
        nn.init.kaiming_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight)
