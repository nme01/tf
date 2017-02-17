import os
import time
from datetime import datetime

import tensorflow as tf

from classification import Classifier
from data_loading import DataLoader
from training import NetTrainer

TMP_DIR = os.path.join('..', 'tmp')
BATCH_SIZE = 1024
TRAIN_LOG_DIR = os.path.join(TMP_DIR, 'summary/train')
MAX_STEPS = 10000


def main():
    weight_decays = [0.05, 0.01] + [0.005]+[0.001, 0.0005]
    lrn_alphas = [0.001, 0.0005]+[0.0001]+[0.00005, 0.00001]

    for weight_decay in weight_decays:
        for lrn_alpha in lrn_alphas:
            clean_log_dir()
            Classifier.WEIGHT_DECAY = weight_decay
            Classifier.LRN_ALPHA = lrn_alpha

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)).as_default() as sess:
                init, loss, train_op, summary_op, validation_accuracy = build_model()
                run_model(init, loss, train_op, sess, summary_op, validation_accuracy)


def clean_log_dir():
    if tf.gfile.Exists(TRAIN_LOG_DIR):
        tf.gfile.DeleteRecursively(TRAIN_LOG_DIR)
    tf.gfile.MakeDirs(TRAIN_LOG_DIR)


def build_model():
    reader = DataLoader(data_dir=TMP_DIR)
    classifier = Classifier()
    trainer = NetTrainer()
    reader.download_dataset_if_necessary()

    train_images, train_labels = reader.load_dataset(batch_size=BATCH_SIZE, use_train_data=True, distort_image=True)
    train_logits = classifier.classify(train_images)
    loss = trainer.loss(train_logits, train_labels)
    train_op = trainer.train(loss)
    training_correct_prediction = tf.nn.in_top_k(train_logits, train_labels, k=1)
    training_accuracy = tf.reduce_mean(tf.cast(training_correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy on training set', training_accuracy)

    eval_images, eval_labels = reader.load_dataset(batch_size=BATCH_SIZE, use_train_data=False, distort_image=False)
    eval_logits = classifier.classify(eval_images, reuse_variables=True)
    validation_correct_prediction = tf.nn.in_top_k(eval_logits, eval_labels, k=1)
    validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy on evaluation set', validation_accuracy)

    init = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()

    return init, loss, train_op, summary_op, validation_accuracy


def run_model(init, loss, train_op, sess, summary_op, validation_accuracy):
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(logdir=TRAIN_LOG_DIR, graph=sess.graph)

    sess.run(init)

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    for step in range(MAX_STEPS):
        start_time = time.time()
        _, loss_value, eval_accuracy_value = sess.run([train_op, loss, validation_accuracy])
        duration = time.time() - start_time

        # if step % 10 == 0:
            # num_examples_per_step = BATCH_SIZE
            # examples_per_sec = num_examples_per_step / duration
            # sec_per_batch = float(duration)
            #
            # log_line = ('{date:s}: step {step:4d}, loss = {loss:.2f} '
            #             '({examples_per_sec:.1f} examples/sec; {sec_per_batch:.3f} sec/batch)').format(
            #     date=str(datetime.now()), step=step, loss=loss_value, examples_per_sec=examples_per_sec,
            #     sec_per_batch=sec_per_batch)
            # print(log_line)

        # if step % 100 == 0:
            # summary_str = sess.run(summary_op)
            # summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0 or (step + 1) == MAX_STEPS:
            # checkpoint_path = os.path.join(TRAIN_LOG_DIR, 'model.chkpt')
            # saver.save(sess, checkpoint_path, global_step=step)
            print('validation accuracy: {eval_accuracy:.2f}'.format(eval_accuracy=eval_accuracy_value))

    coordinator.request_stop()
    coordinator.join(threads)

    summary_writer.close()


if __name__ == '__main__':
    main()
