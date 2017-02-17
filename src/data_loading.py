import os
import sys
import tarfile
import urllib.request as request
from typing import List

import tensorflow as tf
from tensorflow.python.framework import ops


class DataLoader(object):
    """ A class reading the CIFAR-10 data. """

    DATA_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    """ URL to a file containing CIFAR images. """

    CIFAR_DATA_SUBFOLDER_NAME = 'cifar-10-batches-bin'
    """ Name of the folder in all CIFAR data are being stored after extracting the CIFAR archive. """

    IMAGE_SIZE = 24
    """ Size of images being processed (not necessarily original image size). """

    # INFORMATION ABOUT DATA BEING LOADED
    ORIGINAL_IMAGE_SIZE = 32
    """ Size of one side of a square CIFAR-10 image. """

    NUM_OF_COLOR_CHANNELS = 3
    """ Number of channels in the input CIFAR-10 images. All images are RGB images, thus there are 3 channels. """

    IMAGE_BYTES = ORIGINAL_IMAGE_SIZE**2 * NUM_OF_COLOR_CHANNELS
    """ Number of bytes per image. """

    LABEL_BYTES = 1
    """ Number of bytes per label in the input file. """

    RECORD_BYTES = LABEL_BYTES + IMAGE_BYTES
    """ Number of bytes per CIFAR example, which consists of label data followed by the image data. """

    NUM_CLASSES = 10
    """
    Number of classes defined for the problem of classification (for CIFAR-10 it's 10, for CIFAR-100 it's 100).
    """

    TRAIN_NUM_OF_EXAMPLES_PER_EPOCH = 50000
    """
    Number of examples in the training dataset.
    """

    EVAL_NUM_OF_EXAMPLES_PER_EPOCH = 10000
    """
    Number of examples in the evaluation dataset.
    """

    def __init__(self, data_dir: str):
        """
            Parameters
            ----------
            data_dir
                path to directory storing the CIFAR-10 data
        """
        self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
        """
        Defines from how big part of a buffer we will randomly sample while shuffling - bigger means better shuffling
        but slower start up and more memory used.
        """

        self.data_dir = data_dir

    def download_dataset_if_necessary(self):
        """Download and extract the tarball from Alex's website."""
        dest_directory = self.data_dir
        file_name = self.DATA_URL.split('/')[-1]
        file_path = os.path.join(dest_directory, file_name)

        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        if not os.path.exists(file_path):
            file_path = self._download_cifar_data(file_path, file_name)

        tarfile.open(file_path, 'r:gz').extractall(dest_directory)

    def load_dataset(self, batch_size: int, use_train_data: bool, distort_image: bool):
        cifar_folder_path = os.path.join(self.data_dir, self.CIFAR_DATA_SUBFOLDER_NAME)
        if use_train_data:
            file_names = [os.path.join(cifar_folder_path, 'data_batch_{:d}.bin'.format(i)) for i in range(1, 6)]
            num_examples_per_epoch = self.TRAIN_NUM_OF_EXAMPLES_PER_EPOCH
        else:
            file_names = [os.path.join(cifar_folder_path, 'test_batch.bin')]
            num_examples_per_epoch = self.EVAL_NUM_OF_EXAMPLES_PER_EPOCH

        original_image, label = self._load_image_and_label(file_names)

        if distort_image:
            # crop image randomly
            cropped_image = tf.random_crop(original_image, [self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
            preprocessed_image = self._distort_image(cropped_image)
        else:
            # crop image centrally
            preprocessed_image = tf.image.resize_image_with_crop_or_pad(original_image, target_height=self.IMAGE_SIZE,
                                                                        target_width=self.IMAGE_SIZE)

        # normalize so that mean=0 and variance=1
        with ops.name_scope(None, 'standardize_image', [preprocessed_image]):
            standardized_image = tf.image.per_image_standardization(preprocessed_image)

        return self._generate_image_label_batch(standardized_image, label, num_examples_per_epoch, batch_size,
                                                shuffle=distort_image)

    def _download_cifar_data(self, file_path, file_name):
        def _progress(count, block_size, total_size):
            progress = float(count * block_size) / total_size * 100.0
            sys.stdout.write('\r>> Downloading {:s} {:.1f}%'.format(file_name, progress))
            sys.stdout.flush()

        file_path, _ = request.urlretrieve(self.DATA_URL, file_path, _progress)
        statinfo = os.stat(file_path)
        #sys.stdout.write(' '.join(['\nSuccessfully downloaded', file_name, str(statinfo.st_size), 'bytes.']))

        return file_path

    def _load_image_and_label(self, file_names: List[str]):
        with ops.name_scope(None, 'load_image_and_label', [file_names]):
            filename_queue = tf.train.string_input_producer(file_names)

            record_key, record_value = tf.FixedLengthRecordReader(record_bytes=self.RECORD_BYTES).read(filename_queue)
            record_bytes = tf.decode_raw(record_value, tf.uint8)  # convert string to uint8

            label_raw_data = tf.slice(record_bytes, [0], [self.LABEL_BYTES])  # cut out the label raw data
            label = tf.cast(label_raw_data, tf.int32)  # convert the raw data into int32

            # cut out the image raw data
            image_raw_data = tf.slice(record_bytes, [self.LABEL_BYTES], [self.IMAGE_BYTES])

            # image raw data is a series of numbers ordered by color channels by rows by columns.
            # It needs to be reshaped from 1st rank tensor into 3rd rank one
            reshaped_image = tf.reshape(image_raw_data, [self.NUM_OF_COLOR_CHANNELS, self.ORIGINAL_IMAGE_SIZE,
                                                         self.ORIGINAL_IMAGE_SIZE])

            # for convenience we reorder the image so the bytes are ordered by rows then by columns and then by color
            # channels
            transposed_image = tf.transpose(reshaped_image, [1, 2, 0])
            float32_img = tf.cast(transposed_image, tf.float32)

            return float32_img, label

    def _distort_image(self, image):
        with ops.name_scope(None, 'distort_image', [image]):
            distorted_image = tf.image.random_flip_left_right(image)

            # EXPERIMENT with order and parameters of the following operations
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

            return distorted_image

    def _generate_image_label_batch(self, image, label, num_examples_per_epoch, batch_size, shuffle):
        min_queue_examples = int(num_examples_per_epoch * self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE)
        num_preprocess_threads = 16

        # must be larger than min_queue_examples and the amount larger determines the maximum we will prefetch.
        # Recommendation: min_queue_examples + (num_threads + a small safety margin) * batch_size
        queue_capacity = min_queue_examples + 3 * batch_size

        if shuffle:
            images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                         num_threads=num_preprocess_threads, capacity=queue_capacity,
                                                         min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                                 num_threads=num_preprocess_threads, capacity=queue_capacity)

        # display the training images in the visualizer
        tf.summary.image('images', images, max_outputs=15)

        return images, tf.reshape(label_batch, [batch_size])
