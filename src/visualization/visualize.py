import sys
import os
import glob
import csv
import click
import logging
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

from PIL import Image

# Put the main path in the systems path
sys.path.append("/".join(sys.path[0].split("/")[:-2]))

from src.models.train_model import Model
from src.models.train_model import concat_examples
from src.models.predict_model import get_data_info

# ===============================
# General Visualizer class (visc)
# ===============================

class Visualizer(object):
    """
        Visualize the components of a network
    """
    def __init__(self, network):
        """
            Args:
                network (chainer.Link): The trained network to visualize
        """
        self._network = network
        self._bitmap = {}

    def _rescale(self, data):
        """ 
            Rescale the data [0, 255]

            Args:
                data (float[]): the data to rescale
            Returns:
                (int[])
        """
        data -= data.min()
        data /= data.max()
        data *= 255.0
        return data.astype(np.uint8)

    def _get_layer(self, layer_name):
        """
            Get the layer from the network

            Args:
                layer_name (string|chainer.link): name of the layer to visualize or 
                    the layer itself to visualize
        """
        # Get the weight of the filter
        if isinstance(layer_name, basestring):
            return self._network[layer_name]
        else:
            return layer_name


    def plot_filters(self, layer_name, **kwargs):
        """
            Plot the weigths of a layer

            Args:
                layer_name (string|chainer.link): name of the layer to visualize or 
                    the layer itself to visualize
            Returns:
                (pyplot)
        """
        if not self._network[layer_name]:
            raise ValueError("Layer {} does not exists in model")

        # Get the weight of the filter
        layer = self._get_layer(layer_name)

        weights = None
        try:
            weights = layer.W.T
        except:
            weights = layer.W

        bitmaps = [bitmap[0].data for bitmap in weights]
        #bitmaps = np.rollaxis(weights.data, 1, 4)

        # Plot the weigths
        nrow = int(math.sqrt(len(bitmaps))) + 1
        for i in xrange(len(bitmaps)):
            ax = plt.subplot(nrow, nrow, i+1)
            #ax.get_xaxis().set_visible(false)
            #ax.get_yaxis().set_visible(false)
            bitmap = bitmaps[i]
            #bitmap = np.rollaxis(bitmaps[i], 0, 3)
            plt.imshow(self._rescale(bitmap), **kwargs)
        return plt

    def plot_activation(self, layer_name, layer_transformation=None, **kwargs):
        """
            Plot the layer activation (after "activating" a layer with data, e.g: after training/prediction)
            
            Args:
                layer_name (string|chainer.link): name of the layer to visualize or 
                    the layer itself to visualize
                layer_transformation (Function): apply a transformation to the layer before ploting it
            Returns:
                (pyplot)
        """
        layer = self._get_layer(layer_name)

        if layer.data.shape[0] > 1:
            raise ValueError("Can only plot the activation of 1 image not {}".format(layer.data.shape[0]))

        data = None
        if layer_transformation is not None:
            data = layer_transformation(layer)
        else:
            data = layer.data

        # Plot the activation
        nrow = int(math.sqrt(data.shape[1])) + 1
        for i in xrange(data.shape[1]):
            bitmap = data[0][i]
            fmax = np.max(bitmap)
            fmin = np.min(bitmap)

            diff = fmax - fmin if (fmax - fmin) > 0 else 1
            bitmap = ((bitmap - fmin) * 0xff / diff).astype(np.uint8)
            ax = plt.subplot(nrow, nrow, i+1)
            #ax.get_xaxis().set_visible(false)
            #ax.get_yaxis().set_visible(false)
            plt.imshow(bitmap, **kwargs)
        return plt

    def plot_output(self, layer_name, **kwargs):
        """
            Plot the output at a particular layer

            Args:
                layer_name (string|chainer.link): name of the layer to visualize or 
                    the layer itself to visualize
            Returns:
                (pyplot)
        """
        layer = self._get_layer(layer_name)
        output = layer.data

        # Plot the output
        N = layer.shape[0] * layer.shape[1]
        nrow = int(math.sqrt(N)) + 1
        for i in xrange(len(output)):
            for j in xrange(len(output[i])):
                ax = plt.subplot(nrow, nrow, (i) * output.shape[1] + (j+1))
                ax.set_title('Filter: {0}-{1}'.format(i,j), fontsize=10)
                #ax.get_xaxis().set_visible(false)
                #ax.get_yaxis().set_visible(false)
                plt.imshow(output[i][j], **kwargs)
        return plt


# ========================
# Helpers functions (hlpr)
# ========================

def get_coordinates(data, std=[]):
    """
        Extract the coordinate used for plotting for a network

        Args:
            data (float[]): 1D array containing the data to plot
            std (float[]): 1D array to create the "box" arround the curve
        Returns:
            (float[]), (float[]), (float[])
    """
    coord = []
    box = []
    y_min = np.min(data, axis=0)
    y_max = np.max(data, axis=0)

    # Scale the data between range [-1.0, 1.0]
    #data = scale_data(data, mins=y_min, maxs=y_max)

    for i in xrange(len(data)):
        # Create the "box" around the curve
        if len(std) == len(data):
            box.append([data[i] - 1.0 * std[i], data[i] + 1.0 * std[i]])

        coord.append([i, data[i]])

    return np.array(coord, dtype=np.float32), np.array(box, dtype=np.float32), [0, len(coord), y_min, y_max]

def scale_data(data, high=1.0, low=-1.0, maxs=None, mins=None):
    """
        Scale data between [low, high]

        Args:
            data (float[]): 1D array of values to scale
            high (float): upperbound of the scale
            low (float): lowerbound of the scale
            maxs (float): max value in data
            mins (float): min value in data
        Returns:
            (float[])
    """
    if mins is None:
        mins = np.min(data, axis=0)
    if maxs is None:
        maxs = np.max(data, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - data)) / rng)

def plot_data(coordinate, box=[], plt_inst=None, **kwargs):
    """
        Plot the coordinate with the "std box" around the curve

        Args:
            coordinate (float[]): 1D array of the coordinate to plot
            box (float[]): 1D array of the box around the curve
            plt_inst (pyplot): pyplot instance
        Returns:
            (plt_inst)
    """
    if plt_inst is None:
        plt_inst = plt
    
    if len(box) == len(coordinate):
        plt_inst.fill_between(np.arange(len(box)), box[:, 0:1].squeeze(), box[:, 1:].squeeze(), zorder=1, alpha=0.2)

    plt_inst.plot(coordinate[:, 0:1].squeeze(), coordinate[:, 1:].squeeze(), **kwargs)

    return plt_inst

def plot_losses_curves(train_network, valid_network, x_label="Epoch", y_label="Loss", title="Network loss"):
    """
        Plot multiple curves on the same graph

        Args:
            train_network (float[]): the train loss
            valid_network (float[]): the valid loss
            x_label (string): label of x axis
            y_label (string): label of y axis
            title (string): title of the graph
        Returns:
            (plt)
    """
    # Extract the coordinate of the losses
    coord_network_train, box_network_train, stats_network_train = [], [], []
    coord_network_valid, box_network_valid, stats_network_valid = [], [], []
    if len(train_network) > 0:
        coord_network_train, box_network_train, stats_network_train = get_coordinates(train_network[:, 0], train_network[:, 1])
    if len(valid_network) > 0:
        coord_network_valid, box_network_valid, stats_network_valid = get_coordinates(valid_network[:, 0], valid_network[:, 1])

    plt.figure(1)
    plt.subplot("{0}{1}{2}".format(1, 1, 1))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + " (iteration #{})".format(len(coord_network_train) if len(coord_network_train) > 0 else len(coord_network_valid)))
    plt.ylim(
        min(stats_network_train[2] if len(stats_network_train) > 0 else 0, stats_network_valid[2] if len(stats_network_valid) > 0 else 0),
        max(stats_network_train[3] if len(stats_network_train) > 0 else 0, stats_network_valid[3] if len(stats_network_valid) > 0 else 0)
    )

    if len(coord_network_train) > 0:
        plot_data(coord_network_train, box_network_train, plt, label="Train")
    if len(coord_network_valid) > 0:
        plot_data(coord_network_valid, box_network_valid, plt, label="Test")

    plt.legend(ncol=2 if len(coord_network_train) > 0 and len(coord_network_valid) > 0 else 1, loc="upper right", fontsize=10)

    return plt 

def plot(ctx, xaxis, yaxis, title, cb):
    plt.cla()
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(title)
    xcoord = []
    ycoord = []
    for i in xrange(len(ctx)):
        points = cb(ctx[i], i)
        if len(points) != 0:
            xcoord.append(points[0])
            ycoord.append(points[1])

    plt.plot(xcoord, ycoord)
    return plt

def visualize_layer_activation(model, x, layer_idx):
    logger = logging.getLogger(__name__)

    activations = model.activations(layer_idx, x, 0)

    # Rescale the activation [0, 255]
    activations -= activations.min()
    activations /= activations.max()
    activations *= 255
    activations = activations.astype(np.uint8)

    n, c, h, w = activations.shape
    # Plot non-deconvolution image
    #rows = int(math.ceil(math.sqrt(c)))
    #cols = int(round(math.sqrt(c)))
    #plt.figure(1)
    #for i in xrange(c):
    #    plt.subplot(rows, cols, i+1)
    #    plt.imshow(activations[0,i,:,:])
    #return plt


    # Plot deconvolution image
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(round(math.sqrt(n)))
    dpi=100
    scale=1

    plt.figure(1)
    fig, axes = plt.subplots(rows, cols, figsize=(w*cols/dpi*scale, h*rows/dpi*scale), dpi=dpi)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(activations[i].transpose((1, 2, 0)))
    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
    return plt


@click.command()
@click.argument('model', type=click.STRING)
@click.option('--layer_idx', type=click.INT, default=0, help='Convolution layer index.')
@click.option('--model_name', type=click.STRING, default=None, help='Name of the model to visualize.')
@click.option('--data_index', type=click.INT, default=None, help='Index of the data for the visualization.')
@click.option('--model_dir', type=click.Path(exists=True), default='models', help='Directory containing data.')
@click.option('--output_dir', type=click.Path(), default='reports', help='Directory for model checkpoints.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/brain-robotics-data/push/push_testnovel', help='Directory containing data.')
@click.option('--time_step', type=click.INT, default=8, help='Number of time steps to predict.')
@click.option('--model_type', type=click.STRING, default='', help='Type of the trained model.')
@click.option('--schedsamp_k', type=click.FLOAT, default=900.0, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--context_frames', type=click.INT, default=2, help='Number of frames before predictions.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--image_height', type=click.INT, default=64, help='Height of one predicted frame.')
@click.option('--image_width', type=click.INT, default=64, help='Width of one predicted frame.')
def main(model, layer_idx, model_name, data_index, model_dir, output_dir, data_dir, time_step, model_type, schedsamp_k, context_frames, use_state, num_masks, image_height, image_width):
    logger = logging.getLogger(__name__)

    model_path = model_dir + '/' + model
    visualization_path = output_dir + '/' + model
    if not os.path.exists(model_path):
        raise ValueError("Directory {} does not exists".format(model_path))

    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    # @TODO Need to be dynamic reporting
    training_global_losses = None
    if os.path.exists(model_path + '/training-global_losses.npy'): training_global_losses = np.load(model_path + '/training-global_losses.npy')

    training_global_losses_valid = None
    if os.path.exists(model_path + '/training-global_losses_valid.npy'):
        training_global_losses_valid = np.load(model_path + '/training-global_losses_valid.npy')

    #graph = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    #graph.savefig(visualization_path + '/training_global_losses')
    #graph = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses valid', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    #graph.savefig(visualization_path + '/training_global_losses_valid')

    # @TODO: fix the training loss
    #plt_inst = plot_losses_curves(training_global_losses if training_global_losses is not None else [], training_global_losses_valid if training_global_losses_valid is not None else [])
    logger.info("Plotting the loss curves")
    plt_inst = plot_losses_curves(training_global_losses if training_global_losses is not None else [], [])
    iteration_number = len(training_global_losses) if len(training_global_losses) > 0 else len(training_global_losses_valid)
    plt_inst.savefig(visualization_path + "/" + model + "-iteration-{}".format(iteration_number) + ".png")
    plt_inst = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses valid', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    plt_inst.savefig(visualization_path + "/" + model + "-validation-iteration-{}".format(iteration_number) + ".png")

    # Plot the masks activation
    if model_name is not None:
        if not os.path.exists(model_path + '/' + model_name):
            raise ValueError("Model name {} does not exists".format(model_name))

        logger.info("Loading data {}".format(data_index))
        image, image_pred, image_bitmap_pred, action, state = get_data_info(data_dir, data_index)
        img_pred, act_pred, sta_pred = concat_examples([[image_pred, action, state]])

        # Extract the information about the model
        if model_type == '':
            split_name = model.split("-")
            if len(split_name) != 4:
                raise ValueError("Model {} is not recognized, use --model_type to describe the type".format(model))
            model_type = split_name[2]

        # Load the model for prediction
        logger.info("Importing model {0}/{1} of type {2}".format(model_dir, model, model_type))
        pred_model = Model(
            num_masks=num_masks,
            is_cdna=model_type == 'CDNA',
            is_dna=model_type == 'DNA',
            is_stp=model_type == 'STP',
            use_state=use_state,
            scheduled_sampling_k=schedsamp_k,
            num_frame_before_prediction=context_frames,
            prefix='predict'
        )

        chainer.serializers.load_npz(model_path + '/' + model_name, pred_model)
        logger.info("Model imported successfully")
        
        logger.info("Predicting input for the activation map")
        resize_img_pred = []
        for i in xrange(len(img_pred)):
            resize = F.resize_images(img_pred[i], (image_height, image_width))
            resize = F.cast(resize, np.float32) / 255.0
            resize_img_pred.append(resize.data)
        resize_img_pred = np.asarray(resize_img_pred, dtype=np.float32)

        # Only one image to visualize the activation
        plt.cla()

        pred_model([resize_img_pred[0:3], act_pred[0:3], sta_pred[0:3]], 0)
        visualizer = Visualizer(pred_model)

        def deconv(conv):
            def ops(x):
                out_size, in_size, kh, kw = conv.W.data.shape
                #x = L.Deconvolution2D(out_size, in_size, (kh, kw), stride=conv.stride, pad=conv.pad, outsize=(64, 64))(x)
                x = chainer.functions.deconvolution_2d(x, conv.W.data, stride=conv.stride, pad=conv.pad, outsize=(64,64))
                return np.rollaxis(x.data, 1, 4)
            return ops


        #plt_instance = visualizer.plot_activation(model.conv_res[0], deconv(model.enc0))
        logger.info("Creating the layer activation bitmaps")
        for i in xrange(len(pred_model.conv_res)):
            plt.cla()
            plt.figure(1)
            plt_instance = visualizer.plot_activation(pred_model.conv_res[i], interpolation="nearest", cmap="gray")
            plt.savefig(visualization_path + "/" + model + "-iteration-{0}-activation-{1}".format(iteration_number, i) + ".png")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
