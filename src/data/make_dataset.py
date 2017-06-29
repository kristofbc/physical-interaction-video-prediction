# -*- coding: utf-8 -*-

import glob
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import tensorflow as tf
import numpy as np
from PIL import Image
import csv

@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/raw/brain-robotics-data/push/push_train', help='Directory containing data.')
@click.option('--out_dir', type=click.Path(), default='data/processed/brain-robotics-data/push/push_train', help='Output directory of the converted data.')
@click.option('--sequence_length', type=click.INT, default=10, help='Sequence length, including context frames.')
@click.option('--image_original_width', type=click.INT, default=640, help='Original width of the images.')
@click.option('--image_original_height', type=click.INT, default=512, help='Original height of the images.')
@click.option('--image_original_channel', type=click.INT, default=3, help='Original channels amount of the images.')
@click.option('--image_resize_width', type=click.INT, default=64, help='Resize width of the the images.')
@click.option('--image_resize_height', type=click.INT, default=64, help='Resize height of the the images.')
@click.option('--state_action_dimension', type=click.INT, default=5, help='Dimension of the state and action.')
@click.option('--create_img', type=click.INT, default=1, help='Create the bitmap image along the numpy RGB values')
@click.option('--create_img_prediction', type=click.INT, default=1, help='Create the bitmap image used in the prediction phase')
def main(data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    with tf.Session() as sess:
        files = glob.glob(data_dir + '/*')
        if len(files) == 0:
            logger.error("No files found with extensions .tfrecords in directory {0}".format(out_dir))
            exit()

        queue = tf.train.string_input_producer(files, shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        image_seq, state_seq, action_seq = [], [], []
        image_seq_raw = []

        for i in xrange(sequence_length):
            image_name = 'move/' + str(i) + '/image/encoded'
            action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
            state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'

            features = {
                image_name: tf.FixedLenFeature([1], tf.string),
                action_name: tf.FixedLenFeature([state_action_dimension], tf.float32),
                state_name: tf.FixedLenFeature([state_action_dimension], tf.float32)
            }

            features = tf.parse_single_example(serialized_example, features=features)
            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=image_original_channel)
            image.set_shape([image_original_height, image_original_width, image_original_channel])

            # Untouched image used in prediction
            if(create_img_prediction == 1):
                image_pred = tf.identity(image)
                image_pred = tf.reshape(image_pred, [1, image_original_height, image_original_width, image_original_channel])
                image_seq_raw.append(image_pred)

            crop_size = min(image_original_width, image_original_height)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.reshape(image, [1, crop_size, crop_size, image_original_channel])
            # To obtain the original image, with no filter applied to it, comment: reshape, resize_bicubic and cast
            #image = tf.reshape(image, tf.stack([crop_size, crop_size, image_original_channel]))
            image = tf.image.resize_bicubic(image, [image_resize_height, image_resize_width])
            image = tf.cast(image, tf.float32) / 255.0
            image_seq.append(image)

            state = tf.reshape(features[state_name], shape=[1, state_action_dimension])
            state_seq.append(state)
            action = tf.reshape(features[action_name], shape=[1, state_action_dimension])
            action_seq.append(action)

        image_seq = tf.concat(axis=0, values=image_seq)
        state_seq = tf.concat(axis=0, values=state_seq)
        action_seq = tf.concat(axis=0, values=action_seq)
        image_seq_raw = tf.concat(axis=0, values=image_seq_raw)

        #[image_batch, action_batch, state_batch] = tf.train.batch([image_seq, action_seq, state_seq], batch_size, num_threads=batch_size, capacity=100 * batch_size)

        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        logger.info("Saving image_batch, action_batch, state_batch")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        csv_ref = []
        for j in xrange(len(files)):
            logger.info("Creating data from tsrecords {0}/{1}".format(j+1, len(files)))
            raw, act, sta, pred = sess.run([image_seq, action_seq, state_seq, image_seq_raw])
            ref = []
            ref.append(j)

            if create_img == 1:
                for k in xrange(raw.shape[0]):
                    img = Image.fromarray(raw[k], 'RGB')
                    img.save(out_dir + '/image_batch_' + str(j) + '_' + str(k) + '.png')
                ref.append('image_batch_' + str(j) + '_*' + '.png')
            else:
                ref.append('')

            np.save(out_dir + '/image_batch_' + str(j), raw)
            np.save(out_dir + '/action_batch_' + str(j), act)
            np.save(out_dir + '/state_batch_' + str(j), sta)

            ref.append('image_batch_' + str(j) + '.npy')
            ref.append('action_batch_' + str(j) + '.npy')
            ref.append('state_batch_' + str(j) + '.npy')

            # Image used in prediction
            if create_img_prediction == 1:
                np.save(out_dir + '/image_batch_pred_' + str(j), pred)

                for k in xrange(pred.shape[0]):
                    img = Image.fromarray(pred[k], 'RGB')
                    img.save(out_dir + '/image_batch_pred_' + str(j) + '_' + str(k) + '.png')
                ref.append('image_batch_pred_' + str(j) + '_*' + '.png')
                ref.append('image_batch_pred_' + str(j) + '.npy')
            else:
                ref.append('')
                ref.append('')
                
            csv_ref.append(ref)

        logger.info("Writing the results into map file '{0}'".format('map.csv'))
        with open(out_dir + '/map.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['id', 'img_bitmap_path', 'img_np_path', 'action_np_path', 'state_np_path', 'img_bitmap_pred_path', 'img_np_pred_path'])
            for row in csv_ref:
                writer.writerow(row)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
