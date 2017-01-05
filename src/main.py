import logging
import os
import sys

import tensorflow as tf

from cifar_reader import CifarReader

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

# with tf.Session() as sess:
#     summary_writer = tf.train.SummaryWriter("summary", graph_def=sess.graph_def)
reader = CifarReader(data_dir=TMP_DIR)
reader.download_dataset_if_necessary()

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

with tf.Session(config=tf.ConfigProto(log_device_placement=False)).as_default() as sess:
    images, labels = reader.load_dataset(batch_size=BATCH_SIZE, use_train_data=False, distort_image=True)
    summary_op = tf.summary.merge_all()
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=sess.graph)
    summary_result = sess.run(summary_op)
    summary_writer.add_summary(summary_result)
    summary_writer.close()
