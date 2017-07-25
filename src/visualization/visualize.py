import sys
import os
import glob
import csv
import click
import logging

import numpy as np
import matplotlib.pyplot as plt

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

@click.command()
@click.argument('model', type=click.STRING)
@click.option('--model_dir', type=click.Path(exists=True), default='models', help='Directory containing data.')
@click.option('--output_dir', type=click.Path(), default='reports', help='Directory for model checkpoints.')
def main(model, model_dir, output_dir):
    model_path = model_dir + '/' + model
    visualization_path = output_dir + '/' + model
    if not os.path.exists(model_path):
        raise ValueError("Directory {} does not exists".format(model_path))

    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    # @TODO Need to be dynamic reporting
    training_global_losses = None
    if os.path.exists(model_path + '/training-global_losses.npy'):
        training_global_losses = np.load(model_path + '/training-global_losses.npy')

    training_global_losses_valid = None
    if os.path.exists(model_path + '/training-global_losses_valid.npy'):
        training_global_losses_valid = np.load(model_path + '/training-global_losses_valid.npy')

    #graph = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    #graph.savefig(visualization_path + '/training_global_losses')
    #graph = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses valid', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    #graph.savefig(visualization_path + '/training_global_losses_valid')

    # @TODO: fix the training loss
    #plt_inst = plot_losses_curves(training_global_losses if training_global_losses is not None else [], training_global_losses_valid if training_global_losses_valid is not None else [])
    plt_inst = plot_losses_curves(training_global_losses if training_global_losses is not None else [], [])
    iteration_number = len(training_global_losses) if len(training_global_losses) > 0 else len(training_global_losses_valid)
    plt_inst.savefig(visualization_path + "/" + model + "-iteration-{}".format(iteration_number) + ".png")
    plt_inst = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses valid', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    plt_inst.savefig(visualization_path + "/" + model + "-validation-iteration-{}".format(iteration_number) + ".png")

    # Plot the masks activation
    



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
