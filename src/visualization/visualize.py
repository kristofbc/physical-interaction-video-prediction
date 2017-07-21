import sys
import os
import glob
import csv
import click
import logging

import numpy as np
import matplotlib.pyplot as plt

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
    training_global_losses = np.load(model_path + '/training-global_losses.npy')
    training_global_losses_valid = np.load(model_path + '/training-global_losses.npy')

    graph = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    graph.savefig(visualization_path + '/training_global_losses')
    graph = plot(training_global_losses, 'Epoch', 'Mean', 'Training global losses valid', lambda pos, i: [i, pos[0]] if pos[0] != 0 else [] )
    graph.savefig(visualization_path + '/training_global_losses_valid')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
