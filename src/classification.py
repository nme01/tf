import tensorflow as tf

from data_loading import DataLoader


class Classifier(object):
    """ Class used for classifying Cifar images. """

    WEIGHT_DECAY = 0.004
    LRN_ALPHA = 0.0001

    def classify(self, images: tf.Tensor, reuse_variables: bool=False) -> tf.Tensor:
        """
        Parameters
        ----------
        images
            images loaded with CifarReader
        reuse_variables
            whether to use already existing variables or create new ones. Reusing variables is used for evaluating
            the model during the training

        Returns
        -------
        logits for each example (size [batch_size, number_of_classes]).
        """
        with tf.variable_scope('feed_forward', reuse=reuse_variables):
            conv_1 = self._conv(images, filter_edge_length=5, num_of_output_channels=64, name='conv_1',
                                reuse_variables=reuse_variables)
            pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
            norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=Classifier.LRN_ALPHA, beta=0.75, name='norm_1')
            conv_2 = self._conv(norm_1, filter_edge_length=5, num_of_output_channels=64, name='conv_2',
                                reuse_variables=reuse_variables)
            norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=Classifier.LRN_ALPHA, beta=0.75, name='norm_2')
            pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
            fully_connected_1 = self._fully_connected_layer(pool_2, outputs_number=384, name='fully_connected_1',
                                                            reuse_variables=reuse_variables)
            fully_connected_2 = self._fully_connected_layer(fully_connected_1, outputs_number=192,
                                                            name='fully_connected_2', reuse_variables=reuse_variables)
            linear = self._linear(fully_connected_2, reuse_variables=reuse_variables)

        return linear

    def _conv(self, input_tensor, filter_edge_length, num_of_output_channels, name, reuse_variables):
        with tf.variable_scope(name, reuse=reuse_variables) as scope:
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

    def _fully_connected_layer(self, input_tensor, outputs_number, name, reuse_variables):
        with tf.variable_scope(name, reuse=reuse_variables) as scope:
            batch_size = input_tensor.get_shape()[0].value

            # flatten the tensor so that the output can be calculated by simple matrix multiplication
            reshape = tf.reshape(input_tensor, [batch_size, -1])
            num_of_input_values = reshape.get_shape()[1].value

            weights = self._create_variable('weights', shape=[num_of_input_values, outputs_number],
                                            stddev=0.04, weight_decay=Classifier.WEIGHT_DECAY)
            biases = tf.get_variable('biases', [outputs_number], initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

            self._generate_summary(local3)

            return local3

    def _linear(self, input_tensor, reuse_variables):
        with tf.variable_scope('linear', reuse=reuse_variables) as scope:
            inputs_length = input_tensor.get_shape()[-1].value
            weights = self._create_variable('weights', [inputs_length, DataLoader.NUM_CLASSES],
                                            stddev=1/inputs_length, weight_decay=0.0)
            biases = tf.get_variable('biases', [DataLoader.NUM_CLASSES], initializer=tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(input_tensor, weights), biases, name=scope.name)
            self._generate_summary(logits)

            # the activation function is not applied here because the loss function
            # "tf.nn.sparse_softmax_cross_entropy_with_logits" only accepts unscaled logits and applies softmax
            # activation automatically for efficiency
            return logits

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
