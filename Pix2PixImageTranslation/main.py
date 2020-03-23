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
    training_data = DataLoader.PairedImageDataSet(image_folder=args['training_data_folder'],
                                                  batch_size=args['batch_size'],
                                                  shuffle=args['shuffle'],
                                                  preprocess_options=args['PreProcessOptions'],
                                                  load_size=args['load_size'],
                                                  crop_size=args['crop_size'],
                                                  grayscale_input=args['grayscale_input'],
                                                  grayscale_output=args['grayscale_output'])

    validation_data = DataLoader.PairedImageDataSet(image_folder=args['validation_data_folder'],
                                                    batch_size=args['batch_size'],
                                                    shuffle=args['shuffle'],
                                                    preprocess_options=args['PreProcessOptions'],
                                                    load_size=args['load_size'],
                                                    crop_size=args['crop_size'],
                                                    grayscale_input=args['grayscale_input'],
                                                    grayscale_output=args['grayscale_output'])

    test_data = DataLoader.PairedImageDataSet(image_folder=args['test_data_folder'],
                                              batch_size=1,
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
        'device': 'cpu',  # cpu or cuda
        'train': True,  # True for training the network
        'parameter_filename': './parameters/parameters.pt',  # Directory to store the network parameters
        'save_parameters': True,  # Set to True to store the trained
        'lr': 0.0002,  # Learning Rate
        'beta1': (0.5, 0.999),  # Adam Optimizer Parameters
        'lambda_L1': 100.0,  # L1 Regularizer
        'batch_size': 4,  # Batch Size
        'total_epochs': 2,  # Total epochs to train
        'target_real_label': 1.0,  # Numerical value for true label
        'target_forged_label': 0.0,  # Numerical value for false label

        # Data file locations.  Set to None if a validation set is not available
        'training_data_folder': './datasets/facades/train',
        'validation_data_folder': './datasets/facades/val',
        'test_data_folder': './datasets/facades/test',
        'direction': 'B2A',
        'results_folder': './datasets/facades/results',

        # Image processing parameters.  These are specific to the Facade dataset
        'PreProcessOptions': ImageTransforms.PreprocessOptions.RESIZE_AND_CROP,
        'load_size': 286,
        'crop_size': 256,
        'shuffle': True,
        'grayscale_input': False,
        'grayscale_output': False
    }

    # There must be a better way to do the following:
    # --------------------------------------------------------------------------------------------------------------
    # If running the code from CoLab (or Jupyter Notebook):
    #   Comment the code in the next section
    #   Uncomment the lines below
    # --------------------------------------------------------------------------------------------------------------
    # try:
    #     arg = sys.argv[1]
    # except IndexError:
    #     raise SystemExit(f"Usage: {sys.argv[0]} <args_dict>")
    #
    # print(arg)
    # main(arg)
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # If running the code from PyCharm (or any other IDE):
    #   Comment the above section
    #   Uncomment the lines in this section
    # --------------------------------------------------------------------------------------------------------------
    main(args_dict)
    # --------------------------------------------------------------------------------------------------------------

