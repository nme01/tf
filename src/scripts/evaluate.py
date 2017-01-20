import os
import math
from datetime import datetime
import time

import tensorflow as tf
import numpy as np

from classification import Classifier
from data_loading import DataLoader
from scripts.train import TRAIN_LOG_DIR

TMP_DIR = os.path.join('..', 'tmp')
BATCH_SIZE = 1024
EVAL_LOG_DIR = os.path.join(TMP_DIR, 'summary/eval')
NUM_OF_ITERATIONS = int(math.ceil(DataLoader.EVAL_NUM_OF_EXAMPLES_PER_EPOCH / BATCH_SIZE))


def main():
    clean_log_dir()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)).as_default() as sess:
        correct_predictions, summary_op = build_model()
        run_model(sess, correct_predictions, summary_op)


def clean_log_dir():
    if tf.gfile.Exists(EVAL_LOG_DIR):
        tf.gfile.DeleteRecursively(EVAL_LOG_DIR)
    tf.gfile.MakeDirs(EVAL_LOG_DIR)


def build_model():
    reader = DataLoader(data_dir=TMP_DIR)
    classifier = Classifier()
    reader.download_dataset_if_necessary()

    images, labels = reader.load_dataset(batch_size=BATCH_SIZE, use_train_data=True, distort_image=True)
    logits = classifier.classify(images, training=True)
    correct_predictions = tf.nn.in_top_k(logits, labels, k=1)
    summary_op = tf.summary.merge_all()

    return correct_predictions, summary_op


def run_model(sess, correct_predictions, summary_op):
    summary_writer = tf.summary.FileWriter(logdir=EVAL_LOG_DIR, graph=sess.graph)
    saver = tf.train.Saver(tf.global_variables())

    try:
        while True:
            evaluate_net(correct_predictions, saver, sess, summary_op, summary_writer)
            time.sleep(300)  # evaluate the model every 5 minutes
    finally:
        summary_writer.close()


def evaluate_net(correct_predictions, saver, sess, summary_op, summary_writer):
    coord = tf.train.Coordinator()
    chkpt = tf.train.get_checkpoint_state(TRAIN_LOG_DIR)
    if chkpt and chkpt.model_checkpoint_path:
        saver.restore(sess, chkpt.model_checkpoint_path)
        # extract global_step from checkpoint's name
        global_step = chkpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        return
    threads = None
    # noinspection PyBroadException
    try:
        threads = _create_queue_runners(coord, sess)

        step = 0
        correct_total_count = 0
        while step < NUM_OF_ITERATIONS and not coord.should_stop():
            corr_pred = sess.run([correct_predictions])
            correct_total_count += np.sum(corr_pred)
            step += 1

        accuracy = correct_total_count / DataLoader.EVAL_NUM_OF_EXAMPLES_PER_EPOCH
        print('{date:s}: accuracy = {accuracy:.3f}'.format(date=str(datetime.now()), accuracy=accuracy))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Accuracy', simple_value=accuracy)
        summary_writer.add_summary(summary, global_step)

    except Exception as e:
        coord.request_stop(ex=e)
    coord.request_stop()
    coord.join(threads)


def _create_queue_runners(coord, sess):
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    return threads


if __name__ == '__main__':
    main()
