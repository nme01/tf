import logging
import os
import sys

import tensorflow as tf
from tensorflow.python.training.coordinator import Coordinator

from classification import Classifier
from data_loading import DataLoader

TMP_DIR = os.path.join('..', 'tmp')

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]\t %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

BATCH_SIZE = 1000
LOG_DIR = os.path.join(TMP_DIR, 'summary')
MAX_STEPS = 10


def main():
    clean_log_dir()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)).as_default() as sess:
        init, logits, summary_op = build_model()
        run_model(init, logits, sess, summary_op)


def clean_log_dir():
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)


def build_model():
    reader = DataLoader(data_dir=TMP_DIR)
    classifier = Classifier(batch_size=BATCH_SIZE)
    reader.download_dataset_if_necessary()
    images, labels = reader.load_dataset(batch_size=BATCH_SIZE, use_train_data=False, distort_image=True)
    logits = classifier.classify(images)
    init = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()

    return init, logits, summary_op


def run_model(init, logits, sess, summary_op):
    sess.run(init)

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    summary_result = sess.run(summary_op)

    coordinator.request_stop()
    coordinator.join(threads)

    sess.run(logits)

    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=sess.graph)
    summary_writer.add_summary(summary_result)
    summary_writer.close()


if __name__ == '__main__':
    main()
