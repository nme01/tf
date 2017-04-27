import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from classification import Classifier
from train import TMP_DIR
from data_loading import DataLoader


NUM_OF_CHANNELS = 3
patch_size = 7

IMAGE_SIZE = DataLoader.IMAGE_SIZE
NUM_OF_CLASSES = DataLoader.NUM_CLASSES
OCCLUDER_SIZE = 4
BATCH_SIZE = 16


def main():
    images, labels = load_images_and_labels()  # images.shape=[128, 24, 24, 3]
    occluders = generate_occluders()
    heatmaps = create_heatmaps(images, labels, occluders)

    plottable_images = denormalize_images(images)

    # implement proper plotting of heatmaps along original images
    # img = plottable_images[0]
    # plt.imshow(heatmaps[0, :, :].squeeze(), cmap='hot', interpolation='nearest')
    # plt.imshow(img)
    plt.imshow(images[0])
    plt.show()
    # TODO sprawdzić jaka jest różnica pomiędzy oryginalnym obrazkiem a zdenormalizowanym (idealnie by było jakby były identyczne)


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

    heatmaps = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))
    occluded_images_dict = get_occluded_images_dict(images, occluders)
    with tf.Session() as sess:
        images = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
        logits_op = classifier.classify(images, evaluation_mode=False)

        sess.run(tf.global_variables_initializer())

        for top_left_coords in occluded_images_dict.keys():
            occluded_images = occluded_images_dict[top_left_coords]
            occluder = occluders[top_left_coords]

            logits = sess.run(logits_op, feed_dict={images: occluded_images})

            # normalize logits so that they sum up to one across horizontal axis
            logit_sums = np.sum(logits, axis=1)
            logits = logits[np.arange(logits.shape[0]), labels]
            logits = logits / logit_sums

            x_positions = occluder['x_coords']
            y_positions = occluder['y_coords']
            heatmaps[:, x_positions, y_positions] += logits

    heatmaps /= BATCH_SIZE
    return heatmaps


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
    min_max_ranges = max_image_values - min_image_values
    plottable_images = np.array([(images[i, :] - min_image_values[i]) / min_max_ranges[i] for i in range(BATCH_SIZE)])

    return plottable_images


if __name__ == '__main__':
    main()
