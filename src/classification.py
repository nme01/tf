import tensorflow as tf

from data_loading import DataLoader


class Classifier(object):
    """ Class used for classifying Cifar images. """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def classify(self, images: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        images
            Images loaded with CifarReader.

        Returns
        -------
        classes of each image passed
        """
        with tf.variable_scope('feed_forward'):
            conv1 = self._conv(images, filter_edge_length=5, num_of_output_channels=64, name='conv1')
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
            conv2 = self._conv(norm1, filter_edge_length=5, num_of_output_channels=64, name='conv2')
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            local3 = self._fully_connected_layer(pool2, outputs_number=384, name='local3')
            local4 = self._fully_connected_layer(local3, outputs_number=192, name='local4')

            softmax_linear = self._softmax(local4)

        return softmax_linear

    def _conv(self, input_tensor, filter_edge_length, num_of_output_channels, name):
        with tf.variable_scope(name) as scope:
            # last value in input tensor corresponds to the number of input channels in this layer
            input_channels = input_tensor.get_shape()[-1].value
            kernel_shape = [filter_edge_length, filter_edge_length, input_channels, num_of_output_channels]

            kernel = self._create_variable('weights', shape=kernel_shape, stddev=0.05, weight_decay=0.0)
            strides = [1, 1, 1, 1]  # steps for moving the filter

            conv = tf.nn.conv2d(input_tensor, kernel, strides, padding='SAME')
            biases = tf.get_variable('biases', [num_of_output_channels], initializer=tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            self._generate_summary(conv2)

            return conv2

    def _fully_connected_layer(self, input_tensor, outputs_number, name):
        with tf.variable_scope(name) as scope:
            batch_size = input_tensor.get_shape()[0].value

            # flatten the tensor so that the output can be calculated by simple matrix multiplication
            reshape = tf.reshape(input_tensor, [batch_size, -1])
            num_of_input_values = reshape.get_shape()[1].value

            weights = self._create_variable('weights', shape=[num_of_input_values, outputs_number],
                                            stddev=0.04, weight_decay=0.004)
            biases = tf.get_variable('biases', [outputs_number], initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

            self._generate_summary(local3)

            return local3

    def _softmax(self, local4):
        with tf.variable_scope('softmax_linear') as scope:
            inputs_length = local4.get_shape()[-1].value
            weights = self._create_variable('weights', [inputs_length, DataLoader.NUM_CLASSES],
                                            stddev=1/inputs_length, weight_decay=0.0)
            biases = tf.get_variable('biases', [DataLoader.NUM_CLASSES], initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._generate_summary(softmax_linear)

            return softmax_linear

    def _create_variable(self, name, shape, stddev, weight_decay):
        """
        Creates a variable, initializes it and adds a proper weight decay for training purposes. `weight_decay`
        parameter is multiplied by L2Loss weight decay and added to a Tensorflow collection of losses which are used
        in the loss function.
        """
        var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

        if weight_decay is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

    def _generate_summary(self, x):
        tf.summary.histogram(x.op.name + '/activations', x)
        tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))
