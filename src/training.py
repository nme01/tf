import tensorflow as tf


class NetTrainer(object):
    """
    Class responsible for training the model.
    """

    def __init__(self):
        pass

    def train(self):
        pass

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # add all losses from the "losses" collection altogether
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
