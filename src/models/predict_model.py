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
import glob
import subprocess

import six.moves.cPickle as pickle

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops
import imageio

# ========================
# Helpers functions (hlpr)
# ========================

def get_data_info(data_dir, data_index):
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

    image = np.float32(np.load(data_dir + '/' + data_map[data_index][2]))
    image_pred = np.float32(np.load(data_dir + '/' + data_map[data_index][6]))
    image_bitmap_pred = data_map[data_index][5]
    action = np.float32(np.load(data_dir + '/' + data_map[data_index][3]))
    state = np.float32(np.load(data_dir + '/' + data_map[data_index][4]))

    return image, image_pred, image_bitmap_pred, action, state

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
@click.option('--schedsamp_k', type=click.FLOAT, default=-1, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--context_frames', type=click.INT, default=2, help='Number of frames before predictions.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--image_height', type=click.INT, default=64, help='Height of one predicted frame.')
@click.option('--image_width', type=click.INT, default=64, help='Width of one predicted frame.')
@click.option('--original_image_height', type=click.INT, default=512, help='Height of one predicted frame.')
@click.option('--original_image_width', type=click.INT, default=640, help='Width of one predicted frame.')
@click.option('--downscale_factor', type=click.FLOAT, default=0.5, help='Downscale the image by this factor.')
@click.option('--gpu', type=click.INT, default=-1, help='ID of the gpu to use')
@click.option('--gif', type=click.INT, default=1, help='Create a GIF of the predicted result.')
def main(model_dir, model_name, data_index, models_dir, data_dir, time_step, model_type, schedsamp_k, context_frames, use_state, num_masks, image_height, image_width, original_image_height, original_image_width, downscale_factor, gpu, gif):
    """ Predict the next {time_step} frame based on a trained {model} """
    logger = logging.getLogger(__name__)
    path = models_dir + '/' + model_dir
    if not os.path.exists(path + '/' + model_name):
        raise ValueError("Directory {} does not exists".format(path))
    if not os.path.exists(data_dir):
        raise ValueError("Directory {} does not exists".format(data_dir))

    logger.info("Loading data {}".format(data_index))
    image, image_pred, image_bitmap_pred, action, state = get_data_info(data_dir, data_index)

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
        use_state=use_state,
        scheduled_sampling_k=schedsamp_k,
        num_frame_before_prediction=context_frames,
        prefix='predict'
    )

    chainer.serializers.load_npz(path + '/' + model_name, model)
    logger.info("Model imported successfully")

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # Resize the image to fit the trained dimension
    resize_img_pred = []
    for i in xrange(len(img_pred)):
        resize = F.resize_images(img_pred[i], (image_height, image_width))
        resize = F.cast(resize, np.float32) / 255.0
        resize_img_pred.append(resize.data)
    resize_img_pred = np.asarray(resize_img_pred, dtype=np.float32)

    # Predict the new images
    with chainer.using_config('train', False):
        loss = model([resize_img_pred, act_pred, sta_pred], 0)
        predicted_images = model.gen_images

    # Resize the predicted image
    resize_predicted_images = []
    for i in xrange(len(predicted_images)):
        resize = predicted_images[i].data[0]
        resize -= resize.min()
        resize /= resize.max()
        resize *= 255.0
        resize_predicted_images.append(resize.astype(np.uint8))


    # Print the images horizontally
    # First row is the time_step
    # Second row is the ground truth
    # Third row is the generated image
    frame_width = int(original_image_width * downscale_factor)
    frame_height = int(original_image_height * downscale_factor)
    text_width_x = frame_width
    text_height_x = 50
    text_width_y = frame_height
    text_height_y = 50

    total_width = frame_width * time_step + text_height_x
    total_height = frame_height * 2 + text_height_x

    if gif == 1:
        total_width = total_width + frame_width

    new_image = Image.new('RGBA', (total_width, total_height))

    # Text label x
    font_size = 18
    font = ImageFont.truetype('Arial', font_size)
    label = ["Time = {}".format(i+1) for i in xrange(time_step)]

    if gif == 1:
        label.append("Animated sequence")

    for i in xrange(len(label)):
        text = label[i]
        text_container_img = Image.new('RGB', (text_width_x, text_height_x), 'white')
        text_container_draw = ImageDraw.Draw(text_container_img)
        w, h = text_container_draw.textsize(text, font=font)
        text_container_draw.text(((text_width_x-w)/2, (text_height_x-h)/2), text, fill='black', font=font)
        new_image.paste(text_container_img, (text_height_x + text_width_x*i, 0))

    # Text label y
    label = ["Ground truth", "Prediction"]
    for i in xrange(len(label)):
        text = label[i]
        text_container_img = Image.new('RGB', (text_width_y, text_height_y), 'white')
        text_container_draw = ImageDraw.Draw(text_container_img)
        w, h = text_container_draw.textsize(text, font=font)
        text_container_draw.text(((text_width_y-w)/2, (text_height_y-h)/2), text, fill='black', font=font)
        text_container_img = text_container_img.rotate(90, expand=1)
        new_image.paste(text_container_img, (0, text_height_x + text_width_y * i))

    # Original
    ground_truth_images_path = glob.glob(data_dir + '/' + image_bitmap_pred)
    original_gif = []
    for i in xrange(min(time_step, len(ground_truth_images_path))):
        img = Image.open(ground_truth_images_path[i]).convert('RGB')

        if downscale_factor != 1:
            img = img.resize((frame_width, frame_height), Image.ANTIALIAS)

        new_image.paste(img, (text_height_x + frame_width*i, text_height_x))
        #original_gif.append(np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3))
        original_gif.append(img)
    
    # Prediction
    predicted_gif = []
    for i in xrange(len(resize_predicted_images)):
        #img = resize_predicted_images[i].data[0]
        img = resize_predicted_images[i]
        img = np.rollaxis(img, 0, 3)
        img = Image.fromarray(img, 'RGB')

        # Resize the image to the original dimensions
        img = img.resize((original_image_width, original_image_height), Image.ANTIALIAS)

        if downscale_factor != 1:
            img = img.resize((frame_width, frame_height), Image.ANTIALIAS)

        new_image.paste(img, (text_height_x + frame_width*i, frame_height + text_height_x))
        #predicted_gif.append(np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3))
        predicted_gif.append(img)

    # If enabled, create a GIF from the sequence of original and predicted image
    if gif == 1:
        # Create a tmp file
        temp_original_gif_path = path + '/original-' + str(time_step) + model_name + '.gif'
        temp_predicted_gif_path = path + '/predicted-' + str(time_step) + model_name + '.gif'
        #imageio.mimsave(temp_original_gif_path, original_gif)
        #imageio.mimsave(temp_predicted_gif_path, predicted_gif)
        #original_gif_img = Image.open(temp_original_gif_path)
        #predicted_gif_img = Image.open(temp_predicted_gif_path)
        # Import the tmp file and reshape each frame to the whole scene width/height
        gif_frames = []
        for img in original_gif:
            reshaped_original_gif_img = Image.new('RGB', (total_width, total_height))
            reshaped_original_gif_img.paste(img, (text_height_x + frame_width * time_step, text_height_x))
            gif_frames.append(reshaped_original_gif_img)
        for img in predicted_gif:
            reshaped_predicted_gif_img = Image.new('RGB', (total_width, total_height))
            reshaped_predicted_gif_img.paste(img, (text_height_x + frame_width * time_step, text_height_x + frame_height))
            gif_frames.append(reshaped_predicted_gif_img)
        # Avoid flickering when gif is done: add a still under the gif
        new_image.paste(original_gif[0], (text_height_x + frame_width * time_step, text_height_x))
        new_image.paste(predicted_gif[0], (text_height_x + frame_width * time_step, text_height_x + frame_height))
        # Clean the tmp files
        #os.remove(temp_original_gif_path)
        #os.remove(temp_predicted_gif_path)

    if gif == 1:
        new_image.save(path + '/prediction-' + str(time_step) + '-' +  model_name + '.gif', save_all=True, append_images=gif_frames, transparency=0)
    else:
        new_image.save(path + '/prediction-' + str(time_step) + '-' +  model_name + '.png')

    print(model)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
