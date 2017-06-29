#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Predict the next n frames from a trained model
# ==============================================

import numpy as np
import chainer
import chainer.functions as F

from train_model import Model
from train_model import concat_examples

import click
import os
import csv
import logging
from PIL import Image
import six.moves.cPickle as pickle

# =================================================
# Main entry point of the training processes (main)
# =================================================

@click.command()
@click.argument('model_dir', type=click.STRING)
@click.argument('model_name', type=click.STRING)
@click.argument('data_index', type=click.INT)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/brain-robotics-data/push/push_testnovel', help='Directory containing data.')
@click.option('--time_step', type=click.INT, default=8, help='Number of time steps to predict.')
@click.option('--model_type', type=click.STRING, default='', help='Type of the trained model.')
@click.option('--schedsamp_k', type=click.FLOAT, default=900.0, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--context_frames', type=click.INT, default=2, help='Number of frames before predictions.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--image_height', type=click.INT, default=64, help='Height of one predicted frame.')
@click.option('--image_width', type=click.INT, default=64, help='Width of one predicted frame.')
@click.option('--original_image_height', type=click.INT, default=512, help='Height of one predicted frame.')
@click.option('--original_image_width', type=click.INT, default=640, help='Width of one predicted frame.')
@click.option('--gpu', type=click.INT, default=-1, help='ID of the gpu to use')
def main(model_dir, model_name, data_index, models_dir, data_dir, time_step, model_type, schedsamp_k, context_frames, use_state, num_masks, image_height, image_width, original_image_height, original_image_width, gpu):
    """ Predict the next {time_step} frame based on a trained {model} """
    logger = logging.getLogger(__name__)
    path = models_dir + '/' + model_dir
    if not os.path.exists(path + '/' + model_name):
        raise ValueError("Directory {} does not exists".format(path))
    if not os.path.exists(data_dir):
        raise ValueError("Directory {} does not exists".format(data_dir))

    # Get the CSV data map
    data_map = []
    with open(data_dir + '/map.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data_map.append(row)

    if len(data_map) <= 1: # empty or only header
        raise ValueError("No file map found")

    # Get the requested data to test
    data_index = int(data_index)+1
    if data_index > len(data_map)-1:
        raise ValueError("Data index {} is out of range for available data".format(data_index))

    logger.info("Loading data {}".format(data_index))
    image = np.float32(np.load(data_dir + '/' + data_map[data_index][2]))
    image_pred = np.float32(np.load(data_dir + '/' + data_map[data_index][6]))
    action = np.float32(np.load(data_dir + '/' + data_map[data_index][3]))
    state = np.float32(np.load(data_dir + '/' + data_map[data_index][4]))

    img_pred, act_pred, sta_pred = concat_examples([[image_pred, action, state]])

    # Extract the information about the model
    if model_type == '':
        split_name = model_dir.split("-")
        if len(split_name) != 4:
            raise ValueError("Model {} is not recognized, use --model_type to describe the type".format(model_dir))
        model_type = split_name[2]

    # Load the model for prediction
    logger.info("Importing model {0}/{1} of type {2}".format(model_dir, model_name, model_type))
    model = Model(
        num_masks=num_masks,
        is_cdna=model_type == 'CDNA',
        is_dna=model_type == 'DNA',
        is_stp=model_type == 'STP',
        prefix='predict'
    )

    chainer.serializers.load_npz(path + '/' + model_name, model)
    logger.info("Model imported successfully")

    if gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Resize the image to fit the trained dimension
    resize_img_pred = []
    for i in xrange(len(img_pred)):
        resize = F.resize_images(img_pred[i], (image_height, image_width))
        resize = F.cast(resize, np.float32) / 255.0
        resize_img_pred.append(resize.data)
    resize_img_pred = np.asarray(resize_img_pred, dtype=np.float32)

    # Predit the new images
    loss = model(resize_img_pred, act_pred, sta_pred, 0, schedsamp_k, use_state, num_masks, context_frames)
    predicted_images = model.gen_images

    # Resize the predicted image
    resize_predicted_images = []
    for i in xrange(len(predicted_images)):
        resize = predicted_images[i] * 255.0
        resize = F.resize_images(resize, (original_image_height, original_image_width)) 
        resize = F.cast(resize, np.int8)
        resize_predicted_images.append(resize)

    # Print the images horizontally
    total_width = original_image_width * time_step
    total_height = original_image_height
    new_image = Image.new('RGB', (total_width, total_height))
    
    ground_truth_images = resize_predicted_images
    for i in xrange(len(ground_truth_images)):
        img = ground_truth_images[i].data[0]
        img = np.rollaxis(img, 0, 3)
        new_image.paste(Image.fromarray(img, 'RGB'), (original_image_width*i, 0))

    new_image.save(path + '/prediction-' + str(time_step) + '.png')

    print(model)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
