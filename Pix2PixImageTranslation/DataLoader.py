import os.path
import torch.utils.data
from PIL import Image

import ImageTransforms


class PairedImageDataSet(torch.utils.data.Dataset):
    __IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
                        '.TIF', '.tiff', '.TIFF']

    def __init__(self, image_folder, batch_size=4, image_channels=3, shuffle=False,
                 preprocess_options=ImageTransforms.PreprocessOptions.RESIZE_AND_CROP,
                 load_size=286, crop_size=256, grayscale_input=False, grayscale_output=False):

        super(PairedImageDataSet).__init__()

        self.image_location = os.path.normpath(image_folder)
        self.batch_size = batch_size
        self.image_channels = image_channels
        self.shuffle = shuffle
        self.preprocess_options = preprocess_options
        self.load_size = load_size
        self.crop_size = crop_size
        self.grayscale_input = grayscale_input
        self.grayscale_output = grayscale_output

        self.data_files = self.__fetch_images(self)
        self.max_dataset_size = len(self.data_files)

        self.dataloader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

    @staticmethod
    def __is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.__IMG_EXTENSIONS)

    @staticmethod
    def __fetch_images(self):
        images = []
        assert os.path.isdir(self.image_location), '%s is not a valid directory'

        for root, _, filenames in sorted(os.walk(self.image_location)):
            for filename in filenames:
                if self.__is_image_file(self, filename):
                    path = os.path.join(root, filename)
                    images.append(path)
        return sorted(images[:len(images)])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        image_filepath = self.data_files[index]
        AB = Image.open(image_filepath).convert('RGB')

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)

        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = ImageTransforms.get_preprocess_params(self.preprocess_options, A.size, self.load_size,
                                                                 self.crop_size) if self.preprocess_options != ImageTransforms.PreprocessOptions.NONE else None

        A_transform = ImageTransforms.get_transform(self.preprocess_options,
                                                    transform_params,
                                                    self.load_size,
                                                    self.crop_size, self.image_channels, self.grayscale_input)
        B_transform = ImageTransforms.get_transform(self.preprocess_options, transform_params, self.load_size,
                                                    self.crop_size, self.image_channels, self.grayscale_output)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': image_filepath, 'B_paths': image_filepath}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.max_dataset_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data


class ThreeImageDataSet(torch.utils.data.Dataset):
    __IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    def __init__(self, image_folder, batch_size=4, image_channels=4, shuffle=False,
                 preprocess_options=ImageTransforms.PreprocessOptions.RESIZE_AND_CROP,
                 load_size=286, crop_size=256, grayscale_input=False, grayscale_output=False):

        super(ThreeImageDataSet).__init__()

        self.image_location = os.path.normpath(image_folder)
        self.batch_size = batch_size
        self.image_channels = image_channels
        self.shuffle = shuffle
        self.preprocess_options = preprocess_options
        self.load_size = load_size
        self.crop_size = crop_size
        self.grayscale_input = grayscale_input
        self.grayscale_output = grayscale_output

        self.data_files = self.__fetch_images(self)
        self.max_dataset_size = len(self.data_files)

        self.dataloader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

    @staticmethod
    def __is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.__IMG_EXTENSIONS)

    @staticmethod
    def __fetch_images(self):
        images = []
        assert os.path.isdir(self.image_location), '%s is not a valid directory'

        for root, _, filenames in sorted(os.walk(self.image_location)):
            for filename in filenames:
                if self.__is_image_file(self, filename):
                    path = os.path.join(root, filename)
                    images.append(path)
        return sorted(images[:len(images)])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, C, paths
            A (tensor) -- an image in the input domain
            B (tensor) -- a new material
            C (tensor) -- an image rendered in the material
            A_paths (str) -- image paths
        """
        # read a image given a random integer index
        image_filepath = self.data_files[index]

        ABC = Image.open(image_filepath).convert('RGB') if self.image_channels == 3 else Image.open(image_filepath).convert('RGBA')

        # split ABC image into A, B, and C
        w, h = ABC.size
        w2 = int(w / 3)
        w3 = 2 * w2

        A = ABC.crop((0, 0, w2, h))
        B = ABC.crop((w2, 0, w3, h))
        C = ABC.crop((w3, 0, w, h))

        # apply the same transform to A and C.  B is material and we do not want to apply any transforms to it.
        transform_params = ImageTransforms.get_preprocess_params(self.preprocess_options, A.size, self.load_size,
                                                                 self.crop_size) if self.preprocess_options != ImageTransforms.PreprocessOptions.NONE else None

        A_transform = ImageTransforms.get_transform(self.preprocess_options, transform_params, self.load_size,
                                                    self.crop_size, self.image_channels, self.grayscale_input)

        B_transform = ImageTransforms.get_transform(ImageTransforms.PreprocessOptions.NONE, None, self.load_size,
                                                    self.crop_size, self.image_channels, self.grayscale_output)

        C_transform = ImageTransforms.get_transform(self.preprocess_options, transform_params, self.load_size,
                                                    self.crop_size, self.image_channels, self.grayscale_output)

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)

        return {'A': A, 'B': B, 'C': C, 'A_paths': image_filepath}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.max_dataset_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data