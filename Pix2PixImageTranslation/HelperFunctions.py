from enum import Enum
import torch.nn as nn


class NormType(Enum):
    NONE = 0,
    BATCH_NORM = 1,
    INSTANCE_NORM = 2


class ActivationType(Enum):
    RELU = 1,
    LEAKY_RELU = 2,
    SIGMOID = 3,
    TANH = 4


class BlockType(Enum):
    ENCODER = 1,
    DECODER = 2,


def create_block(block_type, in_channels, out_channels, kernel_size, stride, padding, norm_type, activation_type, dropout):
    convolution = None
    activation = None

    if block_type == BlockType.ENCODER:
        convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    elif block_type == BlockType.DECODER:
        convolution = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        raise Exception('ImageTranslationModel::Unknown Block Type')

    if activation_type == ActivationType.LEAKY_RELU:
        activation = nn.LeakyReLU(0.2, True)
    elif activation_type == ActivationType.RELU:
        activation = nn.ReLU(True)
    elif activation_type == ActivationType.SIGMOID:
        activation = nn.Sigmoid()
    elif activation_type == ActivationType.TANH:
        activation = nn.Tanh()
    else:
        raise Exception('ImageTranslationModel::Unknown Activation Type')

    if dropout:
        if norm_type == NormType.NONE:
            block = [convolution, activation, nn.Dropout2d(0.5, True)]

        elif norm_type == NormType.BATCH_NORM:
            block = [convolution, nn.BatchNorm2d(out_channels), activation, nn.Dropout2d(0.5, True)]

        elif norm_type == NormType.INSTANCE_NORM:
            block = [convolution, nn.InstanceNorm2d(out_channels), activation, nn.Dropout2d(0.5, True)]
        else:
            raise Exception('ImageTranslationModel::Unknown Norm Type')
    else:
        if norm_type == NormType.NONE:
            block = [convolution, activation]

        elif norm_type == NormType.BATCH_NORM:
            block = [convolution, nn.BatchNorm2d(out_channels), activation]

        elif norm_type == NormType.INSTANCE_NORM:
            block = [convolution, nn.InstanceNorm2d(out_channels), activation]

    return block
