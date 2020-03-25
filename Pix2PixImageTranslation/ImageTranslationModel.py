"""
ImageTranslationModel.py

Faisal Habib
March 5, 2020

Description:
An implementation of the pix2pix Image Translation
Original Paper:  https://arxiv.org/pdf/1611.07004.pdf

Many of the sections in this file are adopted from
pix2pix_model.py that is available at
https://github.com/phillipi/pix2pix
"""
import time
import os.path

import numpy as np

import torch
import torch.nn as nn

import Pix2PixGenerator
import Pix2PixDiscriminator

import UtilityFunctions
from tqdm import tqdm

from torchsummary import summary


class ImageTranslationModel:
    """
    Implements pix2pix Image Translation
    
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, options):
        self.options = options

        if self.options['device'] == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if self.options['train']:
            self.generator = Pix2PixGenerator.Generator()
            self.discriminator = Pix2PixDiscriminator.Discriminator()

            # self.real_label = torch.tensor(self.options['target_real_label'], dtype=torch.double)
            # self.forged_label = torch.tensor(self.options['target_forged_label'], dtype=torch.double)

            self.real_label = torch.tensor(self.options['target_real_label'])
            self.forged_label = torch.tensor(self.options['target_forged_label'])

            # define loss functions
            self.criterionGAN = nn.BCEWithLogitsLoss()
            self.criterionL1 = nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.options['lr'],
                                                betas=self.options['beta1'])
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.options['lr'],
                                                betas=self.options['beta1'])

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, self.options['lr_decay_step'], self.options['lr_decay_gamma'])
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, self.options['lr_decay_step'], self.options['lr_decay_gamma'])

            # self.discriminator_loss = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
            # self.generator_loss = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

            self.discriminator_loss = torch.tensor(0.0, requires_grad=True)
            self.generator_loss = torch.tensor(0.0, requires_grad=True)

            self.generator = self.generator.train()
            self.discriminator = self.discriminator.train()

            if self.options['device'] == 'cuda':
                self.generator.cuda()
                self.discriminator.cuda()
        else:
            self.generator = torch.load(os.path.join(self.options['parameter_filename']))
            self.generator = self.generator.eval()

            if self.options['device'] == 'cuda':
                self.generator.cuda()

    def backpropagate_discriminator(self, target_img, input_img, forged_img):
        input_img_forged_img = torch.cat((input_img, forged_img), 1)
        # prediction_of_forgery = self.discriminator(input_img_forged_img.detach()).to(dtype=torch.double)
        prediction_of_forgery = self.discriminator(input_img_forged_img.detach())
        loss_forgery = self.compute_loss(prediction_of_forgery, False)

        input_img_target_img = torch.cat((input_img, target_img), 1)
        # prediction_of_real = self.discriminator(input_img_target_img).to(dtype=torch.double)
        prediction_of_real = self.discriminator(input_img_target_img)
        loss_real = self.compute_loss(prediction_of_real, True)

        self.discriminator_loss = 0.5 * (loss_forgery + loss_real)
        self.discriminator_loss.backward()

    def backpropagate_generator(self, target_img, input_img, forged_img):
        input_img_forged_img = torch.cat((input_img, forged_img), 1)
        # prediction_of_forgery = self.discriminator(input_img_forged_img).to(dtype=torch.double)
        prediction_of_forgery = self.discriminator(input_img_forged_img)

        loss_generatorGAN = self.compute_loss(prediction_of_forgery, True)
        # loss_generatorL1 = self.criterionL1(forged_img, target_img).to(dtype=torch.double) * self.options['lambda_L1']
        loss_generatorL1 = self.criterionL1(forged_img, target_img) * self.options['lambda_L1']

        # combine loss and calculate gradients
        self.generator_loss = loss_generatorGAN + loss_generatorL1
        self.generator_loss.backward()

    def train(self, training_dataset, validation_dataset, direction='A2B'):

        self.generator = self.generator.train()
        self.discriminator = self.discriminator.train()

        best_validation_loss = 1e6
        training_losses_generator = []
        training_losses_discriminator = []

        validation_losses_generator = []

        for epoch in range(self.options['total_regular_epochs'] + self.options['total_decay_epochs']):
            swatch_start = time.time()
            print(64 * '-')
            print("Epoch:  {0}/{1}".format(epoch + 1, (self.options['total_regular_epochs'] + self.options['total_decay_epochs'])))

            if self.options['verbose']:
                print('G LR = %.7f' % self.optimizer_G.param_groups[0]['lr'])
                print('D LR = %.7f' % self.optimizer_D.param_groups[0]['lr'])

            losses_generator = []
            losses_discriminator = []

            total_batches = len(training_dataset) // self.options['batch_size']
            with tqdm(desc='Training', total=total_batches, leave=False, unit='batch', position=0) as progressBar:
                for i, data in enumerate(training_dataset):
                    input_img = data['A' if direction == 'A2B' else 'B'].to(self.device)
                    target_img = data['B' if direction == 'A2B' else 'A'].to(self.device)

                    # commit forgery: G(input_img) :(
                    forged_img = self.forward(input_img)

                    # Update Discriminator:
                    # 1. Enable backprop for D
                    # 2. Set Discriminator's gradients to zero
                    # 3. Calculate gradients for D
                    # 4. Update Discriminator's weights
                    self.discriminator.set_requires_grad(True)
                    self.optimizer_D.zero_grad()
                    self.backpropagate_discriminator(target_img, input_img, forged_img)
                    self.optimizer_D.step()

                    # Update Generator:
                    # 1. Discriminator requires no gradients when optimizing Generator
                    # 2. Set Generator's gradients to zero
                    # 3. Calculate gradients for Generator
                    # 4. Update Generator's weights
                    self.discriminator.set_requires_grad(False)
                    self.optimizer_G.zero_grad()
                    self.backpropagate_generator(target_img, input_img, forged_img)
                    self.optimizer_G.step()

                    losses_generator.append(self.generator_loss.item())
                    losses_discriminator.append(self.discriminator_loss.item())

                    progressBar.update(1)

            progressBar.close()
            mean_training_loss_generator = np.mean(losses_generator)
            mean_training_loss_discriminator = np.mean(losses_discriminator)

            training_losses_generator.append(mean_training_loss_generator)
            training_losses_discriminator.append(mean_training_loss_discriminator)

            print("\nGenerator Training Loss = {0:.4f}.  Discriminator Training Loss = {1:.4f}".format(
                mean_training_loss_generator, mean_training_loss_discriminator))
            print("Time Elapsed = {0:.2f} s".format(time.time() - swatch_start))

            # At this stage we should check for the validation error by running the generator on the validation set
            losses_generator_validation = []
            with torch.no_grad():
                self.generator = self.generator.eval()
                self.discriminator = self.discriminator.eval()

                for i, data in enumerate(validation_dataset):
                    input_img = data['A' if direction == 'A2B' else 'B'].to(self.device)
                    target_img = data['B' if direction == 'A2B' else 'A'].to(self.device)

                    # commit forgery: G(input_img) :(
                    forged_img = self.forward(input_img)

                    input_img_forged_img = torch.cat((input_img, forged_img), 1)
                    prediction_of_forgery = self.discriminator(input_img_forged_img)

                    loss_generatorGAN = self.compute_loss(prediction_of_forgery, True)
                    loss_generatorL1 = self.criterionL1(forged_img, target_img) * self.options['lambda_L1']

                    losses_generator_validation.append(0.5 * (loss_generatorGAN.item() + loss_generatorL1.item()))

            mean_validation_loss = np.mean(losses_generator_validation)
            best_validation_loss = mean_validation_loss if mean_validation_loss < best_validation_loss else best_validation_loss
            validation_losses_generator.append(mean_validation_loss)
            print("\nValidation Loss = {0:.4f}\tBest Validation Loss = {1:.4f}".format(mean_validation_loss,
                                                                                       best_validation_loss))

            # Update the learning rates
            if epoch >= self.options['total_regular_epochs'] - 1:
                self.scheduler_G.step()
                self.scheduler_D.step()

        # Save the training plots
        UtilityFunctions.save_loss_plot({'Generator': training_losses_generator,
                                         'Discriminator': training_losses_discriminator,
                                         'Validation': validation_losses_generator},
                                        'Training and Validation Losses', './training_losses_generator.pdf')

        # At the end of training, save the generator weights for future evaluation
        if self.options['save_parameters']:
            torch.save(self.generator, self.options['parameter_filename'])

    def test(self, test_dataset, direction='A2B'):

        print(64 * '-')
        print("Generating Test Results...")
        total_tests = len(test_dataset)
        swatch_start = time.time()
        with torch.no_grad():
            self.generator = self.generator.eval()

            with tqdm(desc='Testing', total=total_tests, leave=False, unit='image', position=0) as progressBar:
                for i, data in enumerate(test_dataset):
                    input_img = data['A' if direction == 'A2B' else 'B'].to(self.device)
                    target_img = data['B' if direction == 'A2B' else 'A'].to(self.device)

                    # commit forgery: G(input_img) :(
                    forged_img = self.forward(input_img)

                    # save the results
                    filename = os.path.split(data['A_paths'][0])[1]
                    UtilityFunctions.save_tensor_image(forged_img, "{0}/forged_{1}".format(self.options['results_folder'], filename))
                    UtilityFunctions.save_tensor_image(target_img, "{0}/target_{1}".format(self.options['results_folder'], filename))

                    progressBar.update(1)

        print("Time Elapsed = {0:.2f}".format(time.time() - swatch_start))
        print(64 * '-')

    def print_generator_network_details(self):
        print("Generator Network")
        print(64 * '-')
        summary(self.generator, (3, 256, 256))

    def print_discriminator_network_details(self):
        print("Discriminator Network")
        print(64 * '-')
        summary(self.discriminator, (6, 256, 256))

    def forward(self, x):
        """
        Compute the forward pass of the neural network (using only the generator network)
        :param x: input image (real)
        :return:  output image (fake)
        """

        return self.generator.forward(x)

    def compute_loss(self, prediction, target_is_real):
        """
        Calculate loss given Discriminator's output and ground truth labels.

        :param prediction:  the prediction output from a discriminator
        :param target_is_real: if the ground truth label is for real images or fake images
        :return: the calculated loss.
        """

        target_tensor = self.real_label if target_is_real else self.forged_label

        loss = self.criterionGAN(prediction, target_tensor.expand_as(prediction))

        return loss
