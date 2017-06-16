#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implementation in Chainer of https://github.com/tensorflow/models/tree/master/video_prediction
# ==============================================================================================

import random
import math
from math import floor, log
import numpy as np

import chainer
from chainer import variable
import chainer.functions as F
import chainer.links as L
from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer.functions.math import square
from chainer.functions.math import sum

import sys
import os
import glob
import csv
import click
import logging

# =============================================
# Helpers functions used accross scripts (hlpe)
# =============================================


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """
        Sample batch with specified mix of ground truth and generated data points.

        Args:
            ground_truth_x: tensor of ground-truth data point
            generated_x: tensor of generated data point
            batch_size: batch size
            num_ground_truth: number of ground-truth examples to include in batch
        Returns:
            New batch with num_ground_truth samples from ground_truth_x and the rest from generated_x
    """
    idx = np.arange(int(batch_size))
    np.random.shuffle(idx)
    ground_truth_idx = np.array(np.take(idx, np.arange(num_ground_truth)))
    generated_idx = np.array(np.take(idx, np.arange(num_ground_truth, int(batch_size))))

    reshaped_ground_truth_x = F.reshape(ground_truth_x, (int(batch_size), -1))
    reshaped_genetated_x = F.reshape(generated_x, (int(batch_size), -1))
    ground_truth_examps = np.take(reshaped_ground_truth_x.data, ground_truth_idx, axis=0)
    generated_examps = np.take(reshaped_genetated_x.data, generated_idx, axis=0)

    index_a = np.vstack((ground_truth_idx, np.zeros_like(ground_truth_idx)))
    index_b = np.vstack((generated_idx, np.ones_like(generated_idx)))
    order = np.hstack((index_a, index_b))[:, np.argsort(np.hstack((ground_truth_idx, generated_idx)))]
    stitched = []
    for i in xrange(len(order[0])):
        if order[1][i] == 0:
            pos = np.where(ground_truth_idx == i)
            stitched.append(ground_truth_examps[pos])
            continue
        else:
            pos = np.where(generated_idx == i)
            stitched.append(generated_examps[pos])
            continue
    stitched = np.reshape(stitched, (ground_truth_x.shape[0], ground_truth_x.shape[1], ground_truth_x.shape[2], ground_truth_x.shape[3]))
    return stitched

def peak_signal_to_noise_ratio(true, pred):
    """
        Image quality metric based on maximal signal power vs. power of the noise

        Args:
            true: the ground truth image
            pred: the predicted image
        Returns:
            Peak signal to noise ratio (PSNR)
    """
    return 10.0 * F.log(1.0 / mean_squared_error(true, pred)) / log(10.0)

def mean_squared_error(true, pred):
    """
        L2 distance between tensors true and pred.

        Args:
            true: the ground truth image
            pred: the predicted image
        Returns:
            Mean squared error between the ground truth and the predicted image
    """
    return F.sum(F.square(true - pred) / pred.size)

def conv_2d(inputs, W=None, stride=1, pad=1):
    """
        Create a basic 2d convolution using Chainer's internal function
        
        Args:
            inputs: input Tensor, 4D, batch x channels x height x width
            W: input weight to apply in the convolution
            stride: stride of filter application
            pad: spatial padding width for inputs Tensor
        Returns:
            Output variable of shape (batch x channels x height x width)
    """
    if W is None:
        W_initializer = initializers._get_initializer(None)
        W = variable.Parameter(W_initializer)
    return convolution_2d.convolution_2d(inputs, None, W, stride, pad)

def layer_normalization_conv_2d(inputs):
    batch_size, channels, height, width = inputs.shape[0:4]
    inputs = F.reshape(inputs, (batch_size, -1))
    inputs = L.LayerNormalization()(inputs)
    inputs = F.reshape(inputs, (batch_size, channels, height, width))
    return inputs

# =============
# Models (mdls)
# =============


class BasicConvLSTMCell(chainer.Chain):
    """ Stateless convolutional LSTM, as seen in lstm_op.py from video_prediction model """

    def __init__(self):
        super(BasicConvLSTMCell, self).__init__()

    def __call__(self, inputs, state, num_channels, filter_size=5, forget_bias=1.0):
        """Basic LSTM recurrent network cell, with 2D convolution connctions.

          We add forget_bias (default: 1) to the biases of the forget gate in order to
          reduce the scale of forgetting in the beginning of the training.

          It does not allow cell clipping, a projection layer, and does not
          use peep-hole connections: it is the basic baseline.

          Args:
            inputs: input Tensor, 4D, batch x channels x height x width
            state: state Tensor, 4D, batch x channels x height x width
            num_channels: the number of output channels in the layer.
            filter_size: the shape of the each convolution filter.
            forget_bias: the initial value of the forget biases.

          Returns:
             a tuple of tensors representing output and the new state.
        """ 
        # In Tensorflow: batch x height x width x channels
        # In Chainer: batch x channel x height x width
        # Create a state based on Finn's implementation
        if state is None:
            state_size = (inputs.shape[0], 2*num_channels, inputs.shape[2], inputs.shape[3]) 
            state = self.xp.zeros(state_size, dtype=inputs[0].data.dtype)

        h, c = F.split_axis(state, indices_or_sections=2, axis=1)
        #inputs_h = np.concatenate((inputs, h), axis=1)
        inputs_h = F.concat((inputs, h))

        # Parameters of gates are concatenated into one conv for efficiency
        i_j_f_o = L.Convolution2D(in_channels=inputs_h.shape[1], out_channels=4*num_channels, ksize=(filter_size, filter_size), pad=filter_size/2)(inputs_h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        #i, j, f, o = np.split(ary=i_j_f_o, indices_or_sections=4, axis=1)
        i, j, f, o = F.split_axis(i_j_f_o, indices_or_sections=4, axis=1)

        new_c = c * F.sigmoid(f + forget_bias) + F.sigmoid(i) * F.tanh(j)
        new_h = F.tanh(new_c) * F.sigmoid(o)

        #return new_h, np.concatenate((new_c, new_h), axis=1)
        return new_h, F.concat((new_c, new_h))

class StatelessCDNA(chainer.Chain):
    """
        Build convolutional lstm video predictor using CDNA
        * Because the CDNA does not keep states, it should be passed as a parameter if one wants to continue learning from previous states
    """
    
    def __init__(self):
        super(StatelessCDNA, self).__init__(
            stateless_lstm = BasicConvLSTMCell()
        )

    def __call__(self, images, actions=None, states=None, iter_num=-1.0, scheduled_sampling_k=-1, use_state=True, num_masks=10, num_frame_before_prediction=2):
        """
            Learn through StatelessCDNA. Because it's stateless, we need to feed the state of the previous iteration if one do not one to start anew
            Args:
                images: tensor of ground truth image sequences
                actions: tensor of action sequences
                states: tensor of ground truth state sequences
                iter_num: tensor of the current training iteration (for sched. sampling)
                k: constant used for scheduled sampling. -1 to feed in own prediction.
                use_state: True to include state and action in prediction
                num_masks: the number of different pixel motion predictions (and
                           the number of masks for each of those predictions)
                feeding in own predictions
                num_frame_before_prediction: number of ground truth frames to pass in before
                                             feeding in own predictions
            Returns:
                gen_images: predicted future image frames
                gen_states: predicted future states
        """
        logger = logging.getLogger(__name__)
        batch_size, color_channels, img_height, img_width = images[0].shape[0:4]
        
        # Generated robot states and images
        gen_states, gen_images = [], []
        current_state = states[0]

        if scheduled_sampling_k == -1:
            feedself = True
        else:
            # Scheduled sampling, inverse sigmoid decay
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = math.exp(iter_num / scheduled_sampling_k)
            num_ground_truth = scheduled_sampling_k + num_ground_truth
            num_ground_truth = scheduled_sampling_k / num_ground_truth
            num_ground_truth = int(round(float(batch_size) * num_ground_truth))
            feedself = False

        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None

        for image, action in zip(images[:-1], actions[:-1]):
            # Reuse variables after the first timestep
            reuse = bool(gen_images)

            done_warm_start = len(gen_images) > num_frame_before_prediction - 1
            if feedself and done_warm_start:
                # Feed in generated image
                prev_image = gen_images[-1]
                logger.info("Feed in generated image")
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
                logger.info("Feed in scheduled_sample")
            else:
                # Always feed in ground_truth
                prev_image = variable.Variable(image)
                logger.info("Feed in normal image")

            # Predicted state is always fed back in
            state_action = F.concat((action, current_state), axis=1)

            enc0 = L.Convolution2D(in_channels=3, out_channels=32, ksize=(5, 5), stride=2, pad=5/2)(prev_image)
            # TensorFlow code use layer_normalization for normalize on the output convolution
            # @TODO: Chainer does not do that, we need our own implementation
            enc0 = layer_normalization_conv_2d(enc0)
            
            hidden1, lstm_state1 = self.stateless_lstm(inputs=enc0, state=lstm_state1, num_channels=32)
            hidden1 = layer_normalization_conv_2d(hidden1)
            hidden2, lstm_state2 = self.stateless_lstm(inputs=hidden1, state=lstm_state2, num_channels=32)
            hidden2 = layer_normalization_conv_2d(hidden2)
            enc1 = L.Convolution2D(in_channels=hidden2.shape[1], out_channels=hidden2.shape[2], ksize=(3,3), stride=2, pad=3/2)(hidden2)

            hidden3, lstm_state3 = self.stateless_lstm(inputs=enc1, state=lstm_state3, num_channels=64)
            hidden3 = layer_normalization_conv_2d(hidden3)
            hidden4, lstm_state4 = self.stateless_lstm(inputs=hidden3, state=lstm_state4, num_channels=64)
            hidden4 = layer_normalization_conv_2d(hidden4)
            enc2 = L.Convolution2D(in_channels=hidden4.shape[1], out_channels=hidden4.shape[1], ksize=(3,3), stride=2, pad=3/2)(hidden4)

            # Pass in state and action
            smear = F.reshape(state_action, (int(batch_size), int(state_action.shape[1]), 1, 1))
            smear = F.tile(smear, (1, 1, int(enc2.shape[2]), int(enc2.shape[3])))

            if use_state:
                enc2 = F.concat((enc2, smear), axis=1) # Previously axis=3 but out channel is on axis=1 ? ok!
            enc3 = L.Convolution2D(in_channels=enc2.shape[1], out_channels=hidden4.shape[1], ksize=(1,1), stride=1, pad=1/2)(enc2)
 
            hidden5, lstm_state5 = self.stateless_lstm(inputs=enc3, state=lstm_state5, num_channels=128)
            hidden5 = layer_normalization_conv_2d(hidden5)
            # ** Had to add outsize + pad!
            enc4 = L.Deconvolution2D(in_channels=hidden5.shape[1], out_channels=hidden5.shape[1], ksize=(3,3), stride=2, outsize=(hidden5.shape[2]*2, hidden5.shape[3]*2), pad=3/2)(hidden5)

            hidden6, lstm_state6 = self.stateless_lstm(inputs=enc4, state=lstm_state6, num_channels=64)
            hidden6 = layer_normalization_conv_2d(hidden6)
            # Skip connection
            hidden6 = F.concat((hidden6, enc1), axis=1) # Previously axis=3 but our channel is on axis=1 ? ok!

            # ** Had to add outsize + pad!
            enc5 = L.Deconvolution2D(in_channels=hidden6.shape[1], out_channels=hidden6.shape[1], ksize=(3,3), stride=2, outsize=(hidden6.shape[2]*2, hidden6.shape[3]*2), pad=3/2)(hidden6)
            hidden7, lstm_state7 = self.stateless_lstm(inputs=enc5, state=lstm_state7, num_channels=32)
            hidden7 = layer_normalization_conv_2d(hidden7)
            # Skip connection
            hidden7 = F.concat((hidden7, enc0), axis=1) # Previously axis=3 but our channel is on axis=1 ? ok!

            # ** Had to add outsize + pad!
            enc6 = L.Deconvolution2D(in_channels=hidden7.shape[1], out_channels=hidden7.shape[1], ksize=(3,3), stride=2, outsize=(hidden7.shape[2]*2, hidden7.shape[3]*2), pad=3/2)(hidden7)
            enc6 = layer_normalization_conv_2d(enc6)

            # CDNA specific
            enc7 = L.Deconvolution2D(in_channels=enc6.shape[1], out_channels=3, ksize=(1,1), stride=1)(enc6)
            transformed = list([F.sigmoid(enc7)])

            # CDNA specific
            # Predict kernels using linear function of last layer
            cdna_input = F.reshape(hidden5, (int(batch_size), -1))
            cdna_kerns = L.Linear(in_size=None, out_size=5*5*num_masks)(cdna_input)
            
            # Reshape and normalize
            #cdna_kerns = np.reshape(cdna_kerns, (batch_size, 5, 5, 1, num_masks))
            cdna_kerns = F.reshape(cdna_kerns, (batch_size, 1, 5, 5, num_masks))
            cdna_kerns = F.relu(cdna_kerns - 1e-12) + 1e-12
            norm_factor = sum.sum(cdna_kerns, (1, 2, 3), keepdims=True)

            # The norm factor is broadcasted to match the shape difference
            axis_reshape = 0
            norm_factor_new_shape = tuple([1] * axis_reshape + list(norm_factor.shape) +
                                           [1] * (len(cdna_kerns.shape) - axis_reshape - len(norm_factor.shape)))
            norm_factor = F.reshape(norm_factor, norm_factor_new_shape)
            norm_factor_broadcasted = F.broadcast_to(norm_factor, cdna_kerns.shape)
            cdna_kerns = cdna_kerns / norm_factor_broadcasted

            #cdna_kerns = np.tile(cdna_kerns, (1,1,1, color_channels, 1))
            cdna_kerns = F.tile(cdna_kerns, (1,3,1,1,1))
            cdna_kerns = F.split_axis(cdna_kerns, indices_or_sections=batch_size, axis=0)
            #prev_images = np.split(ary=prev_image, indices_or_sections=batch_size, axis=0)
            prev_images = F.split_axis(prev_image, indices_or_sections=batch_size, axis=0)

            # Transform image
            tmp_transformed = []
            for kernel, preimg in zip(cdna_kerns, prev_images):
                kernel = F.squeeze(kernel)
                if len(kernel.shape) == 3:
                    kernel = kernel[..., np.keepdims]
                #conv = F.depthwise_convolution_2d(preimg, kernel, [1,1,1,1])
                conv = L.DepthwiseConvolution2D(in_channels=preimg.shape[1], channel_multiplier=kernel.shape[3], ksize=(kernel.shape[1], kernel.shape[2]), stride=1, pad=kernel.shape[1]/2)(preimg)
                tmp_transformed.append(conv)
            tmp_transformed = F.concat(tmp_transformed, axis=0)
            tmp_transformed = F.split_axis(tmp_transformed, indices_or_sections=num_masks, axis=1) # Previously axis=3 but our channels are on axis=1 ? ok!
            transformed = transformed + list(tmp_transformed)

            # Masks
            masks = L.Deconvolution2D(in_channels=enc6.shape[1], out_channels=num_masks+1, ksize=(1,1), stride=1)(enc6)
            #masks = np.reshape(F.softmax(np.reshape(masks, (-1, num_masks + 1))), (int(batch_size), num_masks+1, int(img_height), int(img_width))) # Previously num_mask at the end, but our channels are on axis=1
            masks = F.reshape(masks, (-1, num_masks + 1))
            masks = F.softmax(masks)
            masks = F.reshape(masks, (int(batch_size), num_masks+1, int(img_height), int(img_width))) # Previously num_mask at the end, but our channels are on axis=1? ok!
            mask_list = F.split_axis(masks, indices_or_sections=num_masks+1, axis=1) # Previously axis=3 but our channels are on axis=1 ?
            output = F.scale(prev_image, mask_list[0], axis=0)
            for layer, mask in zip(transformed, mask_list[1:]):
                output += F.scale(layer, mask, axis=0)
            gen_images.append(output)

            current_state = L.Linear(in_size=None, out_size=current_state.shape[1])(state_action)
            gen_states.append(current_state)
            logger.info("StatelessCDNA sub-iteration done")

        return gen_images, gen_states

class Model(chainer.Chain):
    """
        This Model wrap other models like CDNA, STP or DNA.
        It calls their training and get the generated images and states, it then compute the losses and other various parameters
    """
    
    def __init__(self, is_cdna=True, is_dna=False, is_stp=False, prefix=None):
        """
            Initialize a CDNA, STP or DNA through this 'wrapper' Model
            Args:
                is_cdna: if the model should be an extension of CDNA
                is_dna: if the model should be an extension of DNA
                is_stp: if the model should be an extension of STP
                prefix: appended to the results to differentiate between training and validation
                learning_rate: learning rate
        """
        super(Model, self).__init__()
        self.prefix = prefix

        self.psnr_all = 0
        self.loss = 0
        self.train_op = 0

        model = None
        if is_cdna:
            model = StatelessCDNA()
        elif is_stp:
            print('STP')
        elif is_dna:
            print('DNA')
        if model is None:
            raise ValueError("No network specified")
        else:
            self.add_link('model', model)

    def __call__(self, images, actions=None, states=None, iter_num=-1.0, scheduled_sampling_k=-1, use_state=True, num_masks=10, num_frame_before_prediction=2):
        """
            Calls the training process
            Args:
                images: an array of Tensor of shape batch x channels x height x width
                actions: an array of Tensor of shape batch x action
                states: an array of Tensor of shape batch x state
                iter_num: iteration (epoch) index
                scheduled_sampling_k: the hyperparameter k for sheduled sampling
                use_state: if the model should use action+state
                num_masks: number of masks
                num_frame_before_prediction: number of frame before prediction
            Returns:
                loss, all the peak signal to noise ratio, summaries
        """
        gen_images, gen_states = self.model(images, actions, states, iter_num, scheduled_sampling_k, use_state, num_masks, num_frame_before_prediction)

        # L2 loss, PSNR for eval
        loss, psnr_all = 0.0, 0.0
        summaries = []
        for i, x, gx in zip(range(len(gen_images)), images[num_frame_before_prediction:], gen_images[num_frame_before_prediction - 1:]):
            x = variable.Variable(x)
            recon_cost = mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(self.prefix + '_recon_cost' + str(i) + ': ' + str(recon_cost.data))
            summaries.append(self.prefix + '_psnr' + str(i) + ': ' + str(psnr_i.data))
            loss += recon_cost

        for i, state, gen_state in zip(range(len(gen_states)), states[num_frame_before_prediction:], gen_states[num_frame_before_prediction - 1:]):
            state = variable.Variable(state)
            state_cost = mean_squared_error(state, gen_state) * 1e-4
            summaries.append(self.prefix + '_state_cost' + str(i) + ': ' + str(state_cost.data))
            loss += state_cost
        
        summaries.append(self.prefix + '_psnr_all: ' + str(psnr_all.data))
        self.psnr_all = psnr_all
        self.loss = loss = loss / np.float32(len(images) - num_frame_before_prediction)
        summaries.append(self.prefix + '_loss: ' + str(loss.data))
        
        return self.loss, self.psnr_all, summaries


# =================================================
# Main entry point of the training processes (main)
# =================================================


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/brain-robotics-data/push/push_train', help='Directory containing data.')
@click.option('--output_dir', type=click.Path(), default='models', help='Directory for model checkpoints.')
@click.option('--event_log_dir', type=click.Path(), default='models', help='Directory for writing summary.')
@click.option('--epoch', type=click.INT, default=100000, help='Number of training iterations.')
@click.option('--pretrained_model', type=click.Path(), default='', help='Filepath of a pretrained model to initialize from.')
@click.option('--sequence_length', type=click.INT, default=10, help='Sequence length, including context frames.')
@click.option('--context_frames', type=click.INT, default=2, help='Number of frames before predictions.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--model_type', type=click.STRING, default='CDNA', help='Model architecture to use - CDNA, DNA, or STP.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--schedsamp_k', type=click.FLOAT, default=900.0, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--train_val_split', type=click.FLOAT, default=0.95, help='The percentage of data to use for the training set, vs. the validation set.')
@click.option('--batch_size', type=click.INT, default=32, help='Batch size for training.')
@click.option('--learning_rate', type=click.FLOAT, default=0.001, help='The base learning rate of the generator.')
@click.option('--gpu', type=click.INT, default=-1, help='ID of the gpu(s) to use')
@click.option('--validation_interval', type=click.INT, default=200, help='How often to run a batch through the validation model')
@click.option('--save_interval', type=click.INT, default=2000, help='How often to save a model checkpoint')
def main(data_dir, output_dir, event_log_dir, epoch, pretrained_model, sequence_length, context_frames, use_state, model_type, num_masks, schedsamp_k, train_val_split, batch_size, learning_rate, gpu, validation_interval, save_interval):
    """ Train the model based on the data saved in ../processed """
    logger = logging.getLogger(__name__)
    logger.info('Training the model')

    logger.info('Model: {}'.format(model_type))
    logger.info('GPU: {}'.format(gpu))
    logger.info('# Minibatch-size: {}'.format(batch_size))
    logger.info('# epoch: {}'.format(epoch))

    logger.info("Fetching the models and inputs")
    data_map = []
    with open(data_dir + '/map.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data_map.append(row)

    if len(data_map) <= 1: # empty or only header
        logger.error("No file map found")
        exit()
        
    # Load the images, actions and states
    images = []
    actions = []
    states = []
    for i in xrange(1, len(data_map)): # Exclude the header
        logger.info("Loading data {0}/{1}".format(i, len(data_map)-1))
        images.append(np.load(data_dir + '/' + data_map[i][2]))
        actions.append(np.load(data_dir + '/' + data_map[i][3]))
        states.append(np.load(data_dir + '/' + data_map[i][4]))

    images = np.asarray(images)
    actions = np.asarray(actions)
    states = np.asarray(states)

    train_val_split_index = int(np.floor(train_val_split * len(images)))
    images_training = np.asarray(images[:train_val_split_index])
    actions_training = np.asarray(actions[:train_val_split_index])
    states_training = np.asarray(states[:train_val_split_index])

    images_validation = np.asarray(images[train_val_split_index:])
    actions_validation = np.asarray(actions[train_val_split_index:])
    states_validation = np.asarray(states[train_val_split_index:])

    print('Data set contain {0}, {1} will be use for training and {2} will be use for validation'.format(len(images)-1, train_val_split_index, len(images)-1-train_val_split_index))

    # Enable GPU support if defined

    # Create the model    
    training_model = Model(
        is_cdna=model_type == 'CDNA',
        is_dna=model_type == 'DNA',
        is_stp=model_type == 'STP',
        prefix='train'
    )
    validation_model = Model(
        is_cdna=model_type == 'CDNA',
        is_dna=model_type == 'DNA',
        is_stp=model_type == 'STP',
        prefix='val'
    )

    # Create the optimizers for the models
    optimizer = chainer.optimizers.Adam(alpha=learning_rate)
    optimizer.setup(training_model) 

    # @TODO: Create some kind of savers to resume training

    # Training
    if gpu > -1:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    # Run training
    # As per Finn's implementation, one epoch is run on one batch size, randomly, but never more than once.
    # At the end of the queue, if the epochs len is not reach, the queue is generated again. 
    training_queue = []
    validation_queue = []
    fill_length_training = batch_size - (len(images_training) % batch_size)
    fill_length_validation = batch_size - (len(images_validation) % batch_size if len(images_validation) > batch_size else batch_size%len(images_validation))
    for itr in xrange(epoch):
        if len(training_queue) == 0:
            # Create random partition for the batches
            training_queue = random.sample(range(len(images_training)), len(images_training)) + random.sample(range(len(images_training)), fill_length_training)
            training_queue = [training_queue[i:i+batch_size] for i in xrange(0, len(images_training), batch_size)]

        # Create batches
        training_range_indexes = training_queue.pop(0)
        img_training_set = []
        act_training_set = []
        sta_training_set = []
        for idx in training_range_indexes:
            img_training_set.append(images_training[idx])
            act_training_set.append(actions_training[idx])
            sta_training_set.append(states_training[idx])
        img_training_set = np.asarray(img_training_set)
        act_training_set = np.asarray(act_training_set)
        sta_training_set = np.asarray(sta_training_set)
    
        # Split the actions, states and images into timestep
        act_training_set = np.split(ary=act_training_set, indices_or_sections=act_training_set.shape[1], axis=1)
        act_training_set = [np.squeeze(act) for act in act_training_set]
        sta_training_set = np.split(ary=sta_training_set, indices_or_sections=sta_training_set.shape[1], axis=1)
        sta_training_set = [np.squeeze(sta) for sta in sta_training_set]
        img_training_set = np.split(ary=img_training_set, indices_or_sections=img_training_set.shape[1], axis=1)
        # Reshape the img training set to a Chainer compatible tensor : batch x channel x height x width instead of Tensorflow's: batch x height x width x channel
        img_training_set = [np.transpose(np.squeeze(img), (0, 3, 1, 2)) for img in img_training_set]

        # Perform training
        logger.info("Begining training for epoch {}".format(str(itr+1)))
        loss, psnr_all, summaries = training_model(img_training_set, act_training_set, sta_training_set, itr, schedsamp_k, use_state, num_masks, context_frames)
        optimizer.update()

        logger.info("{0} {1}".format(str(itr+1), str(loss.data)))
        
        loss_valid, psnr_all_valid, summaries_valid = None, None, []
        if itr % validation_interval == 2:
            if len(validation_queue) == 0:
                # Create random partition for the batches
                # If the length of the validation is < fill_length_validation, make sure to pad it
                pad = fill_length_validation
                validation_queue = random.sample(range(len(images_validation)), len(images_validation))
                while pad > 0:
                    p = min(fill_length_validation, len(images_validation))
                    p = min(p, pad)
                    pad = pad-p
                    validation_queue += random.sample(range(len(images_validation)), p)

                validation_queue = [validation_queue[i:i+batch_size] for i in xrange(0, max(batch_size,len(images_validation)), batch_size)]

            # Create batches
            validation_range_indexes = validation_queue.pop(0)
            img_validation_set = []
            act_validation_set = []
            sta_validation_set = []

            for idx in validation_range_indexes:
                img_validation_set.append(images_validation[idx])
                act_validation_set.append(actions_validation[idx])
                sta_validation_set.append(states_validation[idx])
            img_validation_set = np.asarray(img_validation_set)
            act_validation_set = np.asarray(act_validation_set)
            sta_validation_set = np.asarray(sta_validation_set)

            act_validation_set = np.split(ary=act_validation_set, indices_or_sections=act_validation_set.shape[1], axis=1)
            act_validation_set = [np.squeeze(act) for act in act_validation_set]
            sta_validation_set = np.split(ary=sta_validation_set, indices_or_sections=sta_validation_set.shape[1], axis=1)
            sta_validation_set = [np.squeeze(sta) for sta in sta_validation_set]
            img_validation_set = np.split(ary=img_validation_set, indices_or_sections=img_validation_set.shape[1], axis=1)
            img_validation_set = [np.squeeze(img) for img in img_validation_set]
            # Reshape the img training set to a Chainer compatible tensor : batch x channel x height x width instead of Tensorflow's: batch x height x width x channel
            img_validation_set = [np.transpose(np.squeeze(img), (0, 3, 1, 2)) for img in img_validation_set]

            # Run through validation set
            loss_valid, psnr_all_valid, summaries_valid = validation_model(img_validation_set, act_validation_set, sta_validation_set, itr, schedsamp_k, use_state, num_masks, context_frames)

        if itr % save_interval == 2:
            logger.info('Saving model')
            # @TODO: Save the model

        for summ in summaries:
            logger.info(summ)
        for summ_valid in summaries_valid:
            logger.info(summ_valid)

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
