import torch
import torch.nn as nn

import HelperFunctions as hf


class Discriminator(nn.Module):
    def __init__(self, train=True):
        super(Discriminator, self).__init__()

        self.discriminator = {
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'num_in1_channels': 3,
            'num_in2_channels': 3,
            'num_out_channels': 1,
            'num_blocks': 5,
            'use_dropout': True
        }

        self.network = nn.Sequential(*self.__build_discriminator())

        self.network.apply(hf.initialize_weights)

    def set_requires_grad(self, requires_grad=False):
        for param in self.network.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """
        This method performs the forward operation of the discriminator
        :param x: input
        :return: output of the last layer
        """
        return self.network.forward(x)

    def __build_discriminator(self):
        """
        A private method that builds the discriminator network.

        :return: list of layers
        """
        blocks = hf.create_block(hf.BlockType.ENCODER,
                                 self.discriminator['num_in1_channels'] +
                                 self.discriminator['num_in2_channels'],
                                 64,
                                 self.discriminator['kernel_size'],
                                 self.discriminator['stride'],
                                 self.discriminator['padding'],
                                 hf.NormType.NONE,
                                 hf.ActivationType.LEAKY_RELU,
                                 self.discriminator['use_dropout'])

        blocks.extend(hf.create_block(hf.BlockType.ENCODER,
                                      64,
                                      128,
                                      self.discriminator['kernel_size'],
                                      self.discriminator['stride'],
                                      self.discriminator['padding'],
                                      hf.NormType.BATCH_NORM,
                                      hf.ActivationType.LEAKY_RELU,
                                      self.discriminator['use_dropout']))

        blocks.extend(hf.create_block(hf.BlockType.ENCODER,
                                      128,
                                      256,
                                      self.discriminator['kernel_size'],
                                      self.discriminator['stride'],
                                      self.discriminator['padding'],
                                      hf.NormType.BATCH_NORM,
                                      hf.ActivationType.LEAKY_RELU,
                                      self.discriminator['use_dropout']))

        blocks.extend(hf.create_block(hf.BlockType.ENCODER,
                                      256,
                                      512,
                                      self.discriminator['kernel_size'],
                                      1,
                                      self.discriminator['padding'],
                                      hf.NormType.BATCH_NORM,
                                      hf.ActivationType.LEAKY_RELU,
                                      self.discriminator['use_dropout']))

        blocks.extend(hf.create_block(hf.BlockType.ENCODER,
                                      512,
                                      self.discriminator['num_out_channels'],
                                      self.discriminator['kernel_size'],
                                      1,
                                      self.discriminator['padding'],
                                      hf.NormType.BATCH_NORM,
                                      hf.ActivationType.NONE,
                                      False))
        return blocks
