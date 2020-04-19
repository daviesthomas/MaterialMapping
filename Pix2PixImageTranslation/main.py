"""
main.py

Faisal Habib
March 10, 2020

Description:
This is the main module that sets up the data, instantiates the pix2pix model, and trains it.

Training datasets and parameters can be changed through the args_dict.
"""

import sys
import ImageTranslationModel
import ImageTransforms
import DataLoader
import torch


def main(args):
    if torch.cuda.is_available():
        args['device'] = 'cuda'
        print("Running on GPU: {}.".format(torch.cuda.get_device_name()))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    else:
        args['device'] = 'cpu'
        print("Running on CPU.")
        torch.set_default_tensor_type('torch.FloatTensor')

    if args['verbose']:
        print(args)

    # Load the training, validation, and test data sets
    if args['direction'] == 'ABC':
        training_data = DataLoader.ThreeImageDataSet(image_folder=args['training_data_folder'],
                                                     batch_size=args['batch_size'],
                                                     image_channels=args['image_channels'],
                                                     shuffle=args['shuffle'],
                                                     preprocess_options=args['PreProcessOptions'],
                                                     load_size=args['load_size'],
                                                     crop_size=args['crop_size'],
                                                     grayscale_input=args['grayscale_input'],
                                                     grayscale_output=args['grayscale_output'])

        validation_data = DataLoader.ThreeImageDataSet(image_folder=args['validation_data_folder'],
                                                       batch_size=args['batch_size'],
                                                       image_channels=args['image_channels'],
                                                       shuffle=args['shuffle'],
                                                       preprocess_options=args['PreProcessOptions'],
                                                       load_size=args['load_size'],
                                                       crop_size=args['crop_size'],
                                                       grayscale_input=args['grayscale_input'],
                                                       grayscale_output=args['grayscale_output'])

        test_data = DataLoader.ThreeImageDataSet(image_folder=args['test_data_folder'],
                                                 batch_size=1,
                                                 image_channels=args['image_channels'],
                                                 shuffle=False,
                                                 preprocess_options=ImageTransforms.PreprocessOptions.NONE)
    else:
        training_data = DataLoader.PairedImageDataSet(image_folder=args['training_data_folder'],
                                                      batch_size=args['batch_size'],
                                                      image_channels=args['image_channels'],
                                                      shuffle=args['shuffle'],
                                                      preprocess_options=args['PreProcessOptions'],
                                                      load_size=args['load_size'],
                                                      crop_size=args['crop_size'],
                                                      grayscale_input=args['grayscale_input'],
                                                      grayscale_output=args['grayscale_output'])

        validation_data = DataLoader.PairedImageDataSet(image_folder=args['validation_data_folder'],
                                                        batch_size=args['batch_size'],
                                                        image_channels=args['image_channels'],
                                                        shuffle=args['shuffle'],
                                                        preprocess_options=args['PreProcessOptions'],
                                                        load_size=args['load_size'],
                                                        crop_size=args['crop_size'],
                                                        grayscale_input=args['grayscale_input'],
                                                        grayscale_output=args['grayscale_output'])

        test_data = DataLoader.PairedImageDataSet(image_folder=args['test_data_folder'],
                                                  batch_size=1,
                                                  image_channels=args['image_channels'],
                                                  shuffle=False,
                                                  preprocess_options=ImageTransforms.PreprocessOptions.NONE)
    # Instantiate the pix2pix network
    pix2pix = ImageTranslationModel.ImageTranslationModel(args)

    if args['verbose']:
        print("Training Images: {} | Validation Images: {} | Test Images: {}".format(len(training_data),
                                                                                     len(validation_data),
                                                                                     len(test_data)))
        pix2pix.print_generator_network_details()
        pix2pix.print_discriminator_network_details()

    # Train the network
    pix2pix.train(training_data, validation_data, args['direction'])

    # Test the network (to be implemented)
    pix2pix.test(test_data, args['direction'])


if __name__ == '__main__':
    # args_dict contains important setup arguments.
    args_dict = {
        'verbose': True,  # Controls print messages.  Set to False to disable printing
        'train': True,  # True for training the network
        'parameter_filename': './parameters/parameters.pt',  # Directory to store the network parameters
        'save_parameters': True,  # Set to True to store the trained
        'lr': 0.0005,  # Learning Rate
        'lr_decay_step': 1,  # Learning Rate decay applied every n step (after the regular epochs training)
        'lr_decay_gamma': 0.90,  # Multiplicative decay rate
        'beta1': (0.5, 0.999),  # Adam Optimizer Parameters
        'lambda_L1': 10.0,  # L1 Regularizer
        'lambda_Style': 0.0,  # Style Regularizer
        'lambda_GP': 0.0,  # Gradient Penalty
        'spectral_norm': True,  # Apply spectral norm to convolution network weights
        'batch_size': 4,  # Batch Size
        'batch_statistics': True,  # Print statistics inbetween batches
        'total_regular_epochs': 10,  # Total epochs to train
        'total_decay_epochs': 10,  # Additional epochs to train with learning rate would decay
        'target_real_label': 1.0,  # Numerical value for true label
        'target_forged_label': 0.0,  # Numerical value for false label

        # Data file locations.  Set to None if a validation set is not available
        'training_data_folder': './datasets/shapes/train',
        'validation_data_folder': './datasets/shapes/val',
        'test_data_folder': './datasets/shapes/test',
        'direction': 'ABC',
        'image_channels': 4,
        'results_folder': './datasets/shapes/results_test_wnbn',

        # Image processing parameters.  These are specific to the Facade dataset
        'PreProcessOptions': ImageTransforms.PreprocessOptions.RESIZE_AND_CROP,
        'load_size': 286,
        'crop_size': 256,
        'shuffle': True,
        'grayscale_input': False,
        'grayscale_output': False
    }

    # --------------------------------------------------------------------------------------------------------------
    main(args_dict)
    # --------------------------------------------------------------------------------------------------------------
