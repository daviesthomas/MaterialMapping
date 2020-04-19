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
import torchvision.transforms as transforms
import torchvision.utils as tu

import StyleLoss
import GradientPenalty

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

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.options['train']:
            self.generator = Pix2PixGenerator.Generator(options['image_channels'],
                                                        options['image_channels'],
                                                        options['spectral_norm'],
                                                        True if options['direction'] == 'ABC' else False)

            self.discriminator = Pix2PixDiscriminator.Discriminator(options['image_channels'],
                                                                    options['image_channels'],
                                                                    options['spectral_norm'],
                                                                    True if options['direction'] == 'ABC' else False)

            self.real_label = torch.tensor(self.options['target_real_label'])
            self.forged_label = torch.tensor(self.options['target_forged_label'])

            # define loss functions
            self.criterionGAN = nn.BCEWithLogitsLoss()
            self.criterionL1 = nn.L1Loss() if self.options['lambda_L1'] > 0.0 else None
            self.styleLoss = StyleLoss.StyleLoss() if self.options['lambda_Style'] > 0.0 else None
            self.gradientPenalty = GradientPenalty.GradientPenalty(self.discriminator) if self.options['lambda_GP'] > 0.0 else None

            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.options['lr'],
                                                betas=self.options['beta1'])
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.options['lr'],
                                                betas=self.options['beta1'])

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, self.options['lr_decay_step'],
                                                               self.options['lr_decay_gamma'])
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, self.options['lr_decay_step'],
                                                               self.options['lr_decay_gamma'])

            self.discriminator_loss = torch.tensor(0.0, requires_grad=True)
            self.generator_loss = torch.tensor(0.0, requires_grad=True)

            self.generator = self.generator.train()
            self.discriminator = self.discriminator.train()

            if self.device == torch.device('cuda'):
                self.generator.cuda()
                self.discriminator.cuda()
        else:
            self.generator = self.generator.eval()
            if self.device == torch.device('cuda'):
                self.generator = torch.load(os.path.join(self.options['parameter_filename']),
                                            map_location=torch.device('cuda'))
                self.generator.cuda()
            else:
                self.generator = torch.load(os.path.join(self.options['parameter_filename']),
                                            map_location=torch.device('cpu'))

    def backpropagate_discriminator(self, target_img, input_img, forged_img, material_img):
        target_img = target_img.to(self.device)
        input_img = input_img.to(self.device)
        forged_img = forged_img.to(self.device)

        material_img = material_img.to(self.device) if material_img is not None else None

        input_img_forged_img = torch.cat((input_img, material_img, forged_img), 1) if material_img is not None else torch.cat((input_img, forged_img), 1)
        input_img_forged_img = input_img_forged_img.detach()

        prediction_of_forgery = self.discriminator(input_img_forged_img)
        loss_forgery = self.compute_loss(prediction_of_forgery, False)



        input_img_target_img = torch.cat((input_img, material_img, target_img), 1) if material_img is not None else torch.cat((input_img, target_img), 1)
        prediction_of_real = self.discriminator(input_img_target_img)
        loss_real = self.compute_loss(prediction_of_real, True)


        gradient_penalty = self.gradientPenalty(forged_img, target_img, input_img, material_img) * self.options['lambda_GP'] if self.options['lambda_GP'] > 0.0 else torch.tensor(0.0)
        self.discriminator_loss = 0.5 * (loss_forgery + loss_real) + gradient_penalty
        self.discriminator_loss.backward()

    def backpropagate_generator(self, target_img, input_img, forged_img, material_img):
        target_img = target_img.to(self.device)
        input_img = input_img.to(self.device)
        forged_img = forged_img.to(self.device)

        material_img = material_img.to(self.device) if material_img is not None else None

        input_img_forged_img = torch.cat((input_img, material_img, forged_img), 1) if material_img is not None else torch.cat((input_img, forged_img), 1)

        prediction_of_forgery = self.discriminator(input_img_forged_img)

        loss_generatorGAN = self.compute_loss(prediction_of_forgery, True)
        loss_generatorL1 = self.criterionL1(forged_img, target_img) * self.options['lambda_L1'] if self.options['lambda_L1'] > 0.0 else torch.tensor(0.0)
        loss_generatorStyle = self.styleLoss(forged_img, target_img) * self.options['lambda_Style'] if self.options['lambda_Style'] > 0.0 else torch.tensor(0.0)


        # combine loss and calculate gradients
        self.generator_loss = loss_generatorGAN + loss_generatorL1 + loss_generatorStyle
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
            print("Epoch:  {0}/{1}".format(epoch + 1,
                                           (self.options['total_regular_epochs'] + self.options['total_decay_epochs'])))

            if self.options['verbose']:
                print('G LR = %.7f' % self.optimizer_G.param_groups[0]['lr'])
                print('D LR = %.7f' % self.optimizer_D.param_groups[0]['lr'])

            losses_generator = []
            losses_discriminator = []

            batch_losses_generator = []
            batch_losses_discriminator = []

            total_batches = len(training_dataset) // self.options['batch_size']
            with tqdm(desc='Training', total=total_batches, leave=True, unit='batch', position=0) as progressBar:
                for i, data in enumerate(training_dataset):
                    if direction == 'ABC':
                        input_img = data['A'].to(self.device)
                        material_img = data['B'].to(self.device)
                        target_img = data['C'].to(self.device)
                    else:
                        input_img = data['A' if direction == 'A2B' else 'B'].to(self.device)
                        material_img = None
                        target_img = data['B' if direction == 'A2B' else 'A'].to(self.device)

                    # commit forgery: G(input_img) :(
                    forged_img = self.forward(input_img, material_img)

                    # Update Discriminator:
                    # 1. Enable backprop for D
                    # 2. Set Discriminator's gradients to zero
                    # 3. Calculate gradients for D
                    # 4. Update Discriminator's weights
                    self.discriminator.set_requires_grad(True)
                    self.optimizer_D.zero_grad()
                    self.backpropagate_discriminator(target_img, input_img, forged_img, material_img)
                    self.optimizer_D.step()

                    # Update Generator:
                    # 1. Discriminator requires no gradients when optimizing Generator
                    # 2. Set Generator's gradients to zero
                    # 3. Calculate gradients for Generator
                    # 4. Update Generator's weights
                    self.discriminator.set_requires_grad(False)
                    self.optimizer_G.zero_grad()
                    self.backpropagate_generator(target_img, input_img, forged_img, material_img)
                    self.optimizer_G.step()

                    losses_generator.append(self.generator_loss.item())
                    losses_discriminator.append(self.discriminator_loss.item())

                    if self.options['batch_statistics']:
                        batch_losses_generator.append(self.generator_loss.item())
                        batch_losses_discriminator.append(self.discriminator_loss.item())

                        if ((i + 1) % (total_batches // 5)) == 0:
                            mean_g_loss = np.mean(batch_losses_generator)
                            mean_d_loss = np.mean(batch_losses_discriminator)
                            batch_losses_generator = []
                            batch_losses_discriminator = []
                            tqdm.write("G.Loss = {0:.4f}.  D.Loss = {1:.4f}".format(mean_g_loss, mean_d_loss), end='')

                    progressBar.update(1)

            progressBar.close()

            mean_training_loss_generator = np.mean(losses_generator)
            mean_training_loss_discriminator = np.mean(losses_discriminator)

            training_losses_generator.append(mean_training_loss_generator)
            training_losses_discriminator.append(mean_training_loss_discriminator)

            print("\nGenerator Training Loss = {0:.4f}.  Discriminator Training Loss = {1:.4f}".format(mean_training_loss_generator, mean_training_loss_discriminator))
            print("Time Elapsed = {0:.2f} s".format(time.time() - swatch_start))

            # At this stage we should check for the validation error by running the generator on the validation set
            losses_generator_validation = []
            with torch.no_grad():
                self.generator = self.generator.eval()
                self.discriminator = self.discriminator.eval()

                for i, data in enumerate(validation_dataset):
                    if direction == 'ABC':
                        input_img = data['A'].to(self.device)
                        material_img = data['B'].to(self.device)
                        target_img = data['C'].to(self.device)
                    else:
                        input_img = data['A' if direction == 'A2B' else 'B'].to(self.device)
                        material_img = None
                        target_img = data['B' if direction == 'A2B' else 'A'].to(self.device)

                    # commit forgery: G(input_img) :(
                    forged_img = self.forward(input_img, material_img)

                    input_img_forged_img = torch.cat((input_img, material_img, forged_img), 1) if material_img is not None else torch.cat((input_img, forged_img), 1)
                    prediction_of_forgery = self.discriminator(input_img_forged_img)

                    loss_generatorGAN = self.compute_loss(prediction_of_forgery, True)
                    loss_generatorL1 = self.criterionL1(forged_img, target_img) * self.options['lambda_L1'] if self.options['lambda_L1'] > 0.0 else torch.tensor(0.0)
                    loss_generatorStyle = self.styleLoss(forged_img, target_img) * self.options['lambda_Style'] if self.options['lambda_Style'] > 0.0 else torch.tensor(0.0)

                    losses_generator_validation.append(loss_generatorGAN.item() + loss_generatorL1.item() + loss_generatorStyle.item())

            mean_validation_loss = np.mean(losses_generator_validation)
            best_validation_loss = mean_validation_loss if mean_validation_loss < best_validation_loss else best_validation_loss
            validation_losses_generator.append(mean_validation_loss)
            print("\nValidation Loss = {0:.4f}\tBest Validation Loss = {1:.4f}\n".format(mean_validation_loss,
                                                                                         best_validation_loss))
            self.generator = self.generator.train(mode=True)
            self.discriminator = self.discriminator.train(mode=True)

            # Update the learning rates
            if epoch >= self.options['total_regular_epochs'] - 1:
                self.scheduler_G.step()
                self.scheduler_D.step()

        # Save the training plots
        UtilityFunctions.save_loss_plot({'Generator': training_losses_generator,
                                         'Validation': validation_losses_generator}, 'Generator Training Losses', './training_generator_loss.pdf')
        UtilityFunctions.save_loss_plot({'Discriminator': training_losses_discriminator}, 'Discriminator Training Loss', './training_discriminator_loss.pdf')

        # UtilityFunctions.save_loss_plot({'Generator': training_losses_generator,
        #                                 'Discriminator': training_losses_discriminator,
        #                                 'Validation': validation_losses_generator},
        #                                'Training and Validation Losses', './training_losses.pdf')

        # At the end of training, save the generator weights for future evaluation
        if self.options['save_parameters']:
            parameter_folder = os.path.split(self.options['parameter_filename'])
            os.mkdir(parameter_folder[0]) if not os.path.exists(parameter_folder[0]) else None
            torch.save(self.generator, self.options['parameter_filename'])

    def test(self, test_dataset, direction='A2B'):

        print(64 * '-')
        print("Generating Test Results...")
        total_tests = len(test_dataset)

        os.mkdir(self.options['results_folder']) if not os.path.exists(self.options['results_folder']) else None

        inv_norm_transform = transforms.Normalize(tuple(-1.0 for i in range(self.options['image_channels'])),
                                                  tuple(2.0 for i in range(self.options['image_channels'])))
        swatch_start = time.time()
        with torch.no_grad():
            self.generator = self.generator.eval()

            with tqdm(desc='Testing', total=total_tests, leave=False, unit='image', position=0) as progressBar:
                for i, data in enumerate(test_dataset):
                    if direction == 'ABC':
                        input_img = data['A'].to(self.device)
                        material_img = data['B'].to(self.device)
                        target_img = data['C'].to(self.device)
                    else:
                        input_img = data['A' if direction == 'A2B' else 'B'].to(self.device)
                        material_img = None
                        target_img = data['B' if direction == 'A2B' else 'A'].to(self.device)

                    # commit forgery: G(input_img) :(
                    forged_img = self.forward(input_img, material_img)

                    # save the results
                    filename = os.path.split(data['A_paths'][0])[1]
                    tu.save_image(inv_norm_transform(forged_img[0]),
                                  "{0}/forged_{1}".format(self.options['results_folder'], filename))
                    tu.save_image(inv_norm_transform(target_img[0]),
                                  "{0}/target_{1}".format(self.options['results_folder'], filename))

                    progressBar.update(1)

        print("\nTime Elapsed = {0:.2f}".format(time.time() - swatch_start))
        print(64 * '-')

    def test_one_image(self, filename, input_image_tensor, material_image_tensor=None, direction='A2B', num_channels=3):

        inv_norm_transform = transforms.Normalize(tuple(-1.0 for i in range(self.options['image_channels'])),
                                                  tuple(2.0 for i in range(self.options['image_channels'])))

        with torch.no_grad():
            self.generator = self.generator.eval()

            input_img = input_image_tensor.to(self.device)
            material_img = material_image_tensor.to(self.device) if material_image_tensor is not None else None

            # commit forgery: G(input_img) :(
            forged_img = self.forward(input_img, material_img)

            # save the results
            tu.save_image(inv_norm_transform(forged_img[0]), filename)

    def print_generator_network_details(self):
        print("Generator Network")
        print(64 * '-')

        if self.options['direction'] == 'ABC':
            summary(self.generator,
                    [(self.options['image_channels'], 256, 256), (self.options['image_channels'], 256, 256)])
        else:
            summary(self.generator, (self.options['image_channels'], 256, 256))

    def print_discriminator_network_details(self):
        print("Discriminator Network")
        print(64 * '-')

        if self.options['direction'] == 'ABC':
            summary(self.discriminator, (3 * self.options['image_channels'], 256, 256))
        else:
            summary(self.discriminator, (2 * self.options['image_channels'], 256, 256))

    def forward(self, x, m):
        """
        Compute the forward pass of the neural network (using only the generator network)
        :param x: input image (real)
        :param m: material image
        :return:  output image (fake)
        """

        return self.generator.forward(x, m)

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
