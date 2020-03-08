"""
ImageTranslationModel.py

Faisal Habib
March 5, 2020

Description:
An implementation of the pix2pix Image Translation
Original Paper:  https://arxiv.org/pdf/1611.07004.pdf
"""

import torch
import torch.nn as nn

import Pix2PixGenerator
import Pix2PixDiscriminator

from torchsummary import summary

class ImageTranslationModel(nn.Module):
    """
    Implements pix2pix Image Translation
    
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, train=True):
        super().__init__()

        self.hardware_message = ""

        torch.seed()
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.hardware_message = "Running on GPU: {}.".format(torch.cuda.get_device_name())
        else:
            self.hardware_message = "Running on CPU."

        if train:
            self.generator = Pix2PixGenerator.Generator()
            self.discriminator = Pix2PixDiscriminator.Discriminator()

    def get_hardware_message(self):
        """
        CPU / GPU Information
        :return: CPU / GPU string message
        """
        return self.hardware_message

    def print_generator_network_details(self):
        print("Generator Network")
        print(80 * '-')
        summary(self.generator, (3,256,256))

    def print_discriminator_network_details(self):
        print("Discriminator Network")
        print(80 * '-')
        summary(self.discriminator, (6, 256, 256))

    def forward(self, x):
        """
        Compute the forward pass of the neural network (using only the generator network)
        :param x: input image (real)
        :return:  output image (fake)
        """

        return self.generator_forward(x)

    def get_loss(self):
        """
        Calculate the loss
        """
