from os.path import join

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from classification import Classifier
from train import TMP_DIR, TRAIN_LOG_DIR
from data_loading import DataLoader


NUM_OF_CHANNELS = 3
patch_size = 7

IMAGE_SIZE = DataLoader.IMAGE_SIZE
NUM_OF_CLASSES = DataLoader.NUM_CLASSES
OCCLUDER_SIZE = 10
BATCH_SIZE = 16
NUM_OF_SHOWN_IMAGES_ROWS = 4
NUM_OF_SHOW_IMAGES_COLS = 4


def main():
    images, labels = load_images_and_labels()  # images.shape=[128, 24, 24, 3]
    occluders = generate_occluders()
    heatmaps = create_heatmaps(images, labels, occluders)

    plottable_images = denormalize_images(images)
    plot_images_and_heatmaps(plottable_images, heatmaps, labels)


def load_images_and_labels():
    reader = DataLoader(data_dir=TMP_DIR)
    reader.download_dataset_if_necessary()

    with tf.Session() as sess:
        images_op, labels_op = reader.load_dataset(batch_size=BATCH_SIZE, use_train_data=False, distort_image=False)

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        images, labels = sess.run([images_op, labels_op])

        coordinator.request_stop()
        coordinator.join(threads)

        return images, labels


def generate_occluders():
    occluders = dict()
    for x in range(IMAGE_SIZE - OCCLUDER_SIZE + 1):
        for y in range(IMAGE_SIZE - OCCLUDER_SIZE + 1):
            occluder = {
                'x_coords': [],
                'y_coords': []
            }

            x_coords, y_coords = np.meshgrid(range(x, x + OCCLUDER_SIZE), range(y, y + OCCLUDER_SIZE))
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()

            occluder['x_coords'] = x_coords
            occluder['y_coords'] = y_coords

            occluders[(x, y)] = occluder

    return occluders


def create_heatmaps(images, labels, occluders):
    classifier = Classifier()
    images_ph = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
    logits_op = classifier.classify(images_ph, evaluation_mode=False)

    model_path = join(TRAIN_LOG_DIR, 'model.chkpt-49999')
    # model_path = join(TRAIN_LOG_DIR, 'model.chkpt-143999')
    saver = tf.train.Saver()

    heatmaps = np.empty((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(occluders)))
    heatmaps[:] = np.nan

    occluded_images_dict = get_occluded_images_dict(images, occluders)

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        for i, top_left_coords in enumerate(occluded_images_dict.keys()):
            occluded_images = occluded_images_dict[top_left_coords]
            occluder = occluders[top_left_coords]

            logits = sess.run(logits_op, feed_dict={images_ph: occluded_images})
            probabilities = softmax(logits)

            # select only those probabilities which correspond to proper labels
            probabilities = probabilities[np.arange(logits.shape[0]), labels]
            probabilities = np.reshape(probabilities, (BATCH_SIZE, 1))

            x_positions = occluder['x_coords']
            y_positions = occluder['y_coords']

            for j in range(BATCH_SIZE):
                heatmaps[j, x_positions, y_positions, i] = probabilities[j]

    heatmaps = np.nanmean(heatmaps, axis=3)
    # DEBUG
    # for i in range(BATCH_SIZE-1):
    #     for j in range(i, BATCH_SIZE):
    #         if np.all(heatmaps[0] == heatmaps[1]):
    #             print("Heamaps nr {:d} and {:d} are the same".format(i,j))

    return heatmaps


def softmax(logits):
    max_logits = np.max(logits, axis=1)
    max_logits = np.reshape(max_logits, (BATCH_SIZE, 1))
    max_logits = np.tile(max_logits, (1, NUM_OF_CLASSES))

    probabilities = np.exp(logits - max_logits)

    sum_of_probabilities = np.sum(probabilities, axis=1)
    sum_of_probabilities = np.reshape(sum_of_probabilities, (BATCH_SIZE, 1))
    sum_of_probabilities = np.tile(sum_of_probabilities, (1, NUM_OF_CLASSES))
    probabilities = probabilities / sum_of_probabilities

    return probabilities


def get_occluded_images_dict(images, occluders):
    occluded_images_dict = dict()  # (x_occluders_pos, y_occluders_pos) -> occluded_images_collection

    for (x, y), occluder in occluders.items():
        occluded_images = np.copy(images)

        x_coords = occluder['x_coords']
        y_coords = occluder['y_coords']

        occluded_images[:, x_coords, y_coords, :] = 0

        occluded_images_dict[(x, y)] = occluded_images

    return occluded_images_dict


def denormalize_images(images):
    flattened_images = np.reshape(images, (BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * NUM_OF_CHANNELS))

    max_image_values = np.max(flattened_images, axis=1)
    min_image_values = np.min(flattened_images, axis=1)

    lower_bound_margin = min_image_values + 127  # by how much min value can be expanded so that it reaches -127
    upper_bound_margin = 128 - max_image_values  # by how much max value can be expanded so that it reaches 128

    bool_indices = lower_bound_margin < upper_bound_margin
    min_image_values = np.reshape(min_image_values, (BATCH_SIZE, 1))
    max_image_values = np.reshape(max_image_values, (BATCH_SIZE, 1))

    repeated_min_values = np.tile(min_image_values[bool_indices], (1, IMAGE_SIZE*IMAGE_SIZE*NUM_OF_CHANNELS))
    flattened_images[bool_indices] = flattened_images[bool_indices] * (-127) / repeated_min_values

    repeated_max_values = np.tile(max_image_values[~bool_indices], (1, IMAGE_SIZE*IMAGE_SIZE*NUM_OF_CHANNELS))
    flattened_images[~bool_indices] = flattened_images[~bool_indices] * 128 / repeated_max_values

    flattened_images = flattened_images + 127
    flattened_images = np.maximum(flattened_images, 0)
    flattened_images = np.minimum(flattened_images, 255)

    flattened_images = flattened_images.astype(np.ubyte)

    plottable_images = np.reshape(flattened_images, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))

    return plottable_images


def plot_images_and_heatmaps(plottable_images, heatmaps, labels):
    label_to_title_dict = {
        0: 'samolot',
        1: 'samochód',
        2: 'ptak',
        3: 'kot',
        4: 'jeleń',
        5: 'pies',
        6: 'żaba',
        7: 'koń',
        8: 'statek',
        9: 'ciężarówka'
    }

    assert NUM_OF_SHOW_IMAGES_COLS % 2 == 0

    # divide num_of_cols by two because there is always image and its heatmap
    num_of_cols = NUM_OF_SHOW_IMAGES_COLS / 2  # type: int

    for i in range(0, NUM_OF_SHOWN_IMAGES_ROWS * NUM_OF_SHOW_IMAGES_COLS, 2):
        tmp_iter = int(i/2)
        image = plottable_images[tmp_iter]
        label = labels[tmp_iter]
        heatmap = heatmaps[tmp_iter]

        plt.subplot(NUM_OF_SHOWN_IMAGES_ROWS, NUM_OF_SHOW_IMAGES_COLS, i+1)
        plt.imshow(image)
        title = label_to_title_dict[label]
        plt.title(title)
        remove_ticks_and_labels()

        plt.subplot(NUM_OF_SHOWN_IMAGES_ROWS, NUM_OF_SHOW_IMAGES_COLS, i + 2)
        plt.imshow(heatmap, cmap='hot')
        remove_ticks_and_labels()

    plt.tight_layout()
    plt.show()


def remove_ticks_and_labels():
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    left='off', right='off', labelleft='off')


if __name__ == '__main__':
    main()
