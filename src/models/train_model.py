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
from chainer import serializers
from chainer.functions.math import square
from chainer.functions.activation import lstm

import sys
import os
import time
import glob
import csv
import click
import logging

# Amount to use when lower bounding Variables
RELU_SHIFT = 1e-12

# Kernel size for DNA and CDNA
DNA_KERN_SIZE = 5

# =============================================
# Helpers functions used accross scripts (hlpe)
# =============================================

def concat_examples(batch):
    img_training_set, act_training_set, sta_training_set = [], [], []
    for idx in xrange(len(batch)):
        img_training_set.append(batch[idx][0])
        act_training_set.append(batch[idx][1])
        sta_training_set.append(batch[idx][2])

    img_training_set = np.array(img_training_set)
    act_training_set = np.array(act_training_set)
    sta_training_set = np.array(sta_training_set)

    # Split the actions, states and images into timestep
    act_training_set = np.split(ary=act_training_set, indices_or_sections=act_training_set.shape[1], axis=1)
    act_training_set = [np.squeeze(act, axis=1) for act in act_training_set]
    sta_training_set = np.split(ary=sta_training_set, indices_or_sections=sta_training_set.shape[1], axis=1)
    sta_training_set = [np.squeeze(sta, axis=1) for sta in sta_training_set]
    img_training_set = np.split(ary=img_training_set, indices_or_sections=img_training_set.shape[1], axis=1)
    # Reshape the img training set to a Chainer compatible tensor : batch x channel x height x width instead of Tensorflow's: batch x height x width x channel
    img_training_set = [np.rollaxis(np.squeeze(img, axis=1), 3, 1) for img in img_training_set]

    return np.array(img_training_set), np.array(act_training_set), np.array(sta_training_set)

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

def broadcast_reshape(x, y, axis=0):
    """
        Reshape y to correspond to shape of x

        Args:
            x: the broadcasted
            y: the broadcastee
            axis: where the reshape will be performed
        Results:
            Output variable of same shape of x
    """
    y_shape = tuple([1] * axis + list(y.shape) +
                [1] * (len(x.shape) - axis - len(y.shape)))
    y_t = F.reshape(y, y_shape)
    y_t = F.broadcast_to(y_t, x.shape)
    return y_t

def broadcasted_division(x, y, axis=0):
    """
        Apply a division x/y where y is broadcasted to x to be able to complete the operation
        
        Args:
            x: the numerator
            y: the denominator
            axis: where the reshape will be performed
        Results:
            Output variable of same shape of x
    """
    y_t = broadcast_reshape(x, y, axis)
    return x / y_t

def broadcast_scale(x, y, axis=0):
    """ Apply a multiplicatation x*y where y is broadcasted to x to be able to complete the operation

        Args:
            x: left hand operation
            y: right hand operation
            axis: where the reshape will be performed
        Resuts:
            Output variable of same shape of x
    """
    y_t = broadcast_reshape(x, y, axis)
    return x*y_t


# =============
# Chains (chns)
# =============
class LayerNormalizationConv2D(chainer.Chain):
    def __init__(self):
        super(LayerNormalizationConv2D, self).__init__(
            norm = L.LayerNormalization()
        )
    
    """
        Apply a "layer normalization" on the result of a convolution

        Args:
            inputs: input tensor, 4D, batch x channel x height x width
        Returns:
            Output variable of shape (batch x channels x height x width)
    """
    def __call__(self, inputs):
        batch_size, channels, height, width = inputs.shape[0:4]
        inputs = F.reshape(inputs, (batch_size, -1))
        inputs = self.norm(inputs)
        inputs = F.reshape(inputs, (batch_size, channels, height, width))
        return inputs


# =============
# Models (mdls)
# =============


class BasicConvLSTMCell(chainer.Chain):
    """ Stateless convolutional LSTM, as seen in lstm_op.py from video_prediction model """

    def __init__(self, in_size, out_size):
        super(BasicConvLSTMCell, self).__init__()

        self.in_size = in_size,
        self.out_size = out_size

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

        c, h = F.split_axis(state, indices_or_sections=2, axis=1)

        #inputs_h = np.concatenate((inputs, h), axis=1)
        inputs_h = F.concat((inputs, h), axis=1)

        # Parameters of gates are concatenated into one conv for efficiency
        i_j_f_o = L.Convolution2D(in_channels=inputs_h.shape[1], out_channels=4*num_channels, ksize=(filter_size, filter_size), pad=filter_size/2)(inputs_h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        j, i, f, o = F.split_axis(i_j_f_o, indices_or_sections=4, axis=1)

        new_c = c * F.sigmoid(f + forget_bias) + F.sigmoid(i) * F.tanh(j)
        new_h = F.tanh(new_c) * F.sigmoid(o)

        #return new_h, np.concatenate((new_c, new_h), axis=1)
        return new_h, F.concat((new_c, new_h), axis=1)

class StatelessCDNA(chainer.Chain):
    """
        Build convolutional lstm video predictor using CDNA
        * Because the CDNA does not keep states, it should be passed as a parameter if one wants to continue learning from previous states
    """
    
    def __init__(self, num_masks):
        super(StatelessCDNA, self).__init__(
            enc7 = L.Deconvolution2D(in_channels=64, out_channels=3, ksize=(1,1), stride=1),
            cdna_kerns = L.Linear(in_size=None, out_size=DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks)
        )

        self.num_masks = num_masks

    def __call__(self, lstm_states, encs, hiddens, batch_size, prev_image, num_masks, color_channels):
        """
            Learn through StatelessCDNA.
            Args:
                lstm_states: An array of computed LSTM transformation
                encs: An array of computed transformation
                hiddens: An array of hidden layers
                batch_size: Size of mini batches
                prev_image: The image to transform
                num_masks: Number of masks to apply
                color_channels: Output color channels
            Returns:
                transformed: A list of masks to apply on the previous image
        """
        logger = logging.getLogger(__name__)
        
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7 = lstm_states
        enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
        hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

        # CDNA specific
        enc7 = self.enc7(enc6)
        enc7 = F.relu(enc7)
        transformed = list([F.sigmoid(enc7)])

        # CDNA specific
        # Predict kernels using linear function of last layer
        cdna_input = F.reshape(hidden5, (int(batch_size), -1))
        cdna_kerns = self.cdna_kerns(cdna_input)
        
        # Reshape and normalize
        #cdna_kerns = F.reshape(cdna_kerns, (batch_size, 1, 5, 5, num_masks))
        cdna_kerns = F.reshape(cdna_kerns, (int(batch_size), self.num_masks, 1, DNA_KERN_SIZE, DNA_KERN_SIZE))
        cdna_kerns = F.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = F.sum(cdna_kerns, (2, 3, 4), keepdims=True)

        # The norm factor is broadcasted to match the shape difference
        axis_reshape = 0
        norm_factor_new_shape = tuple([1] * axis_reshape + list(norm_factor.shape) +
                                       [1] * (len(cdna_kerns.shape) - axis_reshape - len(norm_factor.shape)))
        norm_factor = F.reshape(norm_factor, norm_factor_new_shape)
        norm_factor_broadcasted = F.broadcast_to(norm_factor, cdna_kerns.shape)
        cdna_kerns = cdna_kerns / norm_factor_broadcasted

        cdna_kerns = F.tile(cdna_kerns, (1,1,3,1,1))
        cdna_kerns = F.split_axis(cdna_kerns, indices_or_sections=batch_size, axis=0)
        prev_images = F.split_axis(prev_image, indices_or_sections=batch_size, axis=0)

        # Transform image
        tmp_transformed = []
        for kernel, preimg in zip(cdna_kerns, prev_images):
            kernel = F.squeeze(kernel)
            if len(kernel.shape) == 3:
                kernel = kernel[..., np.keepdims]
            if len(preimg.shape) ==3:
                preimg = F.expand_dims(preimg, axis=0)
            conv = F.depthwise_convolution_2d(preimg, kernel, stride=(1, 1), pad=kernel.shape[2]/2)
            tmp_transformed.append(conv)
        tmp_transformed = F.concat(tmp_transformed, axis=0)
        tmp_transformed = F.split_axis(tmp_transformed, indices_or_sections=self.num_masks, axis=1) # Previously axis=3 but our channels are on axis=1 ? ok!
        transformed = transformed + list(tmp_transformed)

        return transformed


class StatelessDNA(chainer.Chain):
    """
        Build convolutional lstm video predictor using DNA
        * Because the DNA does not keep states, it should be passed as a parameter if one wants to continue learning from previous states
    """
    
    def __init__(self, num_masks):
        super(StatelessDNA, self).__init__(
            enc7 = L.Deconvolution2D(in_channels=64, out_channels=DNA_KERN_SIZE**2, ksize=(1,1), stride=1),
            
        )
        self.num_masks = num_masks

    def __call__(self, lstm_states, encs, hiddens, batch_size, prev_image, num_masks, color_channels):
        """
            Learn through StatelessDNA.
            Args:
                lstm_states: An array of computed LSTM transformation
                encs: An array of computed transformation
                hiddens: An array of hidden layers
                batch_size: Size of mini batches
                prev_image: The image to transform
                num_masks: Number of masks to apply
                color_channels: Output color channels
            Returns:
                transformed: A list of masks to apply on the previous image
        """
        logger = logging.getLogger(__name__)
        
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7 = lstm_states
        enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
        hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

        # DNA specific
        enc7 = self.enc7(enc6)
        if num_masks != 1:
            raise ValueError('Only one mask is supported for DNA model.')

        # Construct translated images
        img_height = prev_image.shape[2]
        img_width = prev_image.shape[3]
        prev_image_pad = F.pad(prev_image, pad_width=[[0,0], [0,0], [2,2], [2,2]], mode='constant', constant_values=0)
        kernel_inputs = []
        for xkern in range(DNA_KERN_SIZE):
            for ykern in range(DNA):
                #tmp = F.get_item(prev_image_pad, [prev_image_pad.shape[0], prev_image_pad.shape[0], xkern:img_height, ykern:img_width])
                tmp = F.get_item(prev_image_pad, list([slice(0,prev_image_pad.shape[0]), slice(0,prev_image_pad.shape[1]), slice(xkern,img_height), slice(ykern,img_width)]))
                # ** Added this operation to make sure the size was still the original one!
                tmp = F.pad(tmp, [[0,0], [0,0], [0, xkern], [0, ykern]], mode='constant', constant_values=0)
                tmp = F.expand_dims(tmp, axis=1) # Previously axis=3 but our channel is on axis=1 ? ok!
                kernel_inputs.append(tmp.data)
        kernel_inputs = F.concat(kernel_inputs, axis=1) # Previously axis=3 but our channel us on axis=1 ? ok!

        # Normalize channels to 1
        kernel_normalized = F.relu(enc7 - RELU_SHIFT) + RELU_SHIFT
        kernel_normalized_sum = F.sum(kernel_normalized, axis=1, keepdims=True) # Previously axis=3 but our channel are on axis 1 ? ok!
        kernel_normalized = broadcasted_division(kernel_normalized, kernel_normalized_sum)
        kernel_normalized = F.expand_dims(kernel_normalized, axis=2)
        kernel_normalized = F.scale(kernel_inputs, kernel_normalized, axis=0)
        kernel_normalized = F.sum(kernel_normalized, axis=1, keepdims=False)
        transformed = [kernel_normalized]

        return transformed

class StatelessSTP(chainer.Chain):
    """
        Build convolutional lstm video predictor using STP
        * Because the STP does not keep states, it should be passed as a parameter if one wants to continue learning from previous states
    """
    
    def __init__(self):
        super(StatelessSTP, self).__init__(
            enc7 = L.Deconvolution2D(in_channels=64, out_channels=3, ksize=(1,1), stride=1),
            stp_input = L.Linear(in_size=None, out_size=100),
            params = L.Linear(in_size=None, out_size=6)
        )

    def __call__(self, lstm_states, encs, hiddens, batch_size, prev_image, num_masks, color_channels):
        """
            Learn through StatelessSTP.
            Args:
                lstm_states: An array of computed LSTM transformation
                encs: An array of computed transformation
                hiddens: An array of hidden layers
                batch_size: Size of mini batches
                prev_image: The image to transform
                num_masks: Number of masks to apply
                color_channels: Output color channels
            Returns:
                transformed: A list of masks to apply on the previous image
        """
        logger = logging.getLogger(__name__)
        
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7 = lstm_states
        enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
        hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

        # STP specific
        enc7 = self.enc7(enc6)
        transformed = list([F.sigmoid(enc7)])

        stp_input0 = F.reshape(hidden5, (int(batch_size), -1))
        stp_input1 = self.stp_input(stp_input0)
        identity_params = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        identity_params = np.repeat(identity_params, int(batch_size), axis=0)
        identity_params = variable.Variable(identity_params)

        stp_transformations = []
        for i in range(num_masks-1):
            params = self.params(stp_input1) + identity_params
            params = F.reshape(params, (int(params.shape[0]), 2, 3))
            grid = F.spatial_transformer_grid(params, (prev_image.shape[2], prev_image.shape[3]))
            trans = F.spatial_transformer_sampler(prev_image, grid)
            stp_transformations.append(trans)

        return stp_transformations


class Model(chainer.Chain):
    """
        This Model wrap other models like CDNA, STP or DNA.
        It calls their training and get the generated images and states, it then compute the losses and other various parameters
    """
    
    def __init__(self, num_masks, is_cdna=True, is_dna=False, is_stp=False, prefix=None):
        """
            Initialize a CDNA, STP or DNA through this 'wrapper' Model
            Args:
                is_cdna: if the model should be an extension of CDNA
                is_dna: if the model should be an extension of DNA
                is_stp: if the model should be an extension of STP
                prefix: appended to the results to differentiate between training and validation
                learning_rate: learning rate
        """
        super(Model, self).__init__(
	    enc0 = L.Convolution2D(in_channels=3, out_channels=32, ksize=(5, 5), stride=2, pad=5/2),
            enc1 = L.Convolution2D(in_channels=32, out_channels=32, ksize=(3,3), stride=2, pad=3/2),
            enc2 = L.Convolution2D(in_channels=64, out_channels=64, ksize=(3,3), stride=2, pad=3/2),
            enc3 = L.Convolution2D(in_channels=74, out_channels=64, ksize=(1,1), stride=1, pad=1/2),
            enc4 = L.Deconvolution2D(in_channels=128, out_channels=128, ksize=(3,3), stride=2, outsize=(16,16), pad=3/2),
            enc5 = L.Deconvolution2D(in_channels=96, out_channels=96, ksize=(3,3), stride=2, outsize=(32,32), pad=3/2),
            enc6 = L.Deconvolution2D(in_channels=64, out_channels=64, ksize=(3,3), stride=2, outsize=(64,64), pad=3/2),

            lstm1 = BasicConvLSTMCell(in_size=None, out_size=32),
            lstm2 = BasicConvLSTMCell(in_size=None, out_size=32),
            lstm3 = BasicConvLSTMCell(in_size=None, out_size=64),
            lstm4 = BasicConvLSTMCell(in_size=None, out_size=64),
            lstm5 = BasicConvLSTMCell(in_size=None, out_size=128),
            lstm6 = BasicConvLSTMCell(in_size=None, out_size=64),
            lstm7 = BasicConvLSTMCell(in_size=None, out_size=32),
            
            norm_enc0 = LayerNormalizationConv2D(),
            norm_enc6 = LayerNormalizationConv2D(),
            hidden1 = LayerNormalizationConv2D(),
            hidden2 = LayerNormalizationConv2D(),
            hidden3 = LayerNormalizationConv2D(),
            hidden4 = LayerNormalizationConv2D(),
            hidden5 = LayerNormalizationConv2D(),
            hidden6 = LayerNormalizationConv2D(),
            hidden7 = LayerNormalizationConv2D(),

            masks = L.Deconvolution2D(in_channels=64, out_channels=num_masks+1, ksize=(1,1), stride=1),

            current_state = L.Linear(in_size=None, out_size=5)
	)
        self.num_masks = num_masks
        self.prefix = prefix

        model = None
        if is_cdna:
            model = StatelessCDNA(num_masks)
        elif is_stp:
            model = StatelessSTP(num_masks)
        elif is_dna:
            model = StatelessDNA(num_masks)
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
        logger = logging.getLogger(__name__)
        batch_size, color_channels, img_height, img_width = images[0].shape[0:4]

        #img_training_set = [np.transpose(np.squeeze(img), (0, 3, 1, 2)) for img in img_training_set]
        
        # Generated robot states and images
        gen_states, gen_images = [], []
        current_state = states[0]

        if scheduled_sampling_k == -1:
            feedself = True
        else:
            # Scheduled sampling, inverse sigmoid decay
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = np.int32(
                np.round(np.float32(batch_size) * (scheduled_sampling_k / (scheduled_sampling_k + np.exp(iter_num / scheduled_sampling_k))))
            )
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
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
                prev_image = variable.Variable(prev_image)
            else:
                # Always feed in ground_truth
                prev_image = variable.Variable(image)

            # Predicted state is always fed back in
            state_action = F.concat((action, current_state), axis=1)

            enc0 = self.enc0(prev_image)
            enc0 = F.relu(enc0)
            # TensorFlow code use layer_normalization for normalize on the output convolution
            enc0 = self.norm_enc0(enc0)
            
            hidden1, lstm_state1 = self.lstm1(inputs=enc0, state=lstm_state1, num_channels=32)
            hidden1 = self.hidden1(hidden1)
            hidden2, lstm_state2 = self.lstm2(inputs=hidden1, state=lstm_state2, num_channels=32)
            hidden2 = self.hidden2(hidden2)
            enc1 = self.enc1(hidden2)
            enc1 = F.relu(enc1)

            hidden3, lstm_state3 = self.lstm3(inputs=enc1, state=lstm_state3, num_channels=64)
            hidden3 = self.hidden3(hidden3)
            hidden4, lstm_state4 = self.lstm4(inputs=hidden3, state=lstm_state4, num_channels=64)
            hidden4 = self.hidden4(hidden4)
            enc2 = self.enc2(hidden4)
            enc2 = F.relu(enc2)

            # Pass in state and action
            smear = F.reshape(state_action, (int(batch_size), int(state_action.shape[1]), 1, 1))
            smear = F.tile(smear, (1, 1, int(enc2.shape[2]), int(enc2.shape[3])))

            if use_state:
                enc2 = F.concat((enc2, smear), axis=1) # Previously axis=3 but out channel is on axis=1 ? ok!
            enc3 = self.enc3(enc2)
            enc3 = F.relu(enc3)
 
            hidden5, lstm_state5 = self.lstm5(inputs=enc3, state=lstm_state5, num_channels=128)
            hidden5 = self.hidden5(hidden5)
            # ** Had to add outsize + pad!
            enc4 = self.enc4(hidden5)
            enc4 = F.relu(enc4)

            hidden6, lstm_state6 = self.lstm6(inputs=enc4, state=lstm_state6, num_channels=64)
            hidden6 = self.hidden6(hidden6)
            # Skip connection
            hidden6 = F.concat((hidden6, enc1), axis=1) # Previously axis=3 but our channel is on axis=1 ? ok!

            # ** Had to add outsize + pad!
            enc5 = self.enc5(hidden6)
            enc5 = F.relu(enc5)
            hidden7, lstm_state7 = self.lstm7(inputs=enc5, state=lstm_state7, num_channels=32)
            hidden7 = self.hidden7(hidden7)
            # Skip connection
            hidden7 = F.concat((hidden7, enc0), axis=1) # Previously axis=3 but our channel is on axis=1 ? ok!

            # ** Had to add outsize + pad!
            enc6 = self.enc6(hidden7)
            enc6 = F.relu(enc6)
            enc6 = self.norm_enc6(enc6)

            # Specific model transformations
            transformed = self.model(
                [lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7],
                [enc0, enc1, enc2, enc3, enc4, enc5, enc6],
                [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7],
                batch_size, prev_image, num_masks, int(color_channels)
            )

            # Masks
            masks = self.masks(enc6)
            masks = F.relu(masks)
            masks = F.reshape(masks, (-1, num_masks + 1))
            masks = F.softmax(masks)
            masks = F.reshape(masks, (int(batch_size), num_masks+1, int(img_height), int(img_width))) # Previously num_mask at the end, but our channels are on axis=1? ok!
            mask_list = F.split_axis(masks, indices_or_sections=num_masks+1, axis=1) # Previously axis=3 but our channels are on axis=1 ?
            #output = F.scale(prev_image, mask_list[0], axis=0)
            output = broadcast_scale(prev_image, mask_list[0], axis=0)
            for layer, mask in zip(transformed, mask_list[1:]):
                #output += F.scale(layer, mask, axis=0)
                output += broadcast_scale(layer, mask, axis=0)
            gen_images.append(output)

            current_state = self.current_state(state_action)
            gen_states.append(current_state)

        # End of transformations

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
            print(recon_cost.data)

        for i, state, gen_state in zip(range(len(gen_states)), states[num_frame_before_prediction:], gen_states[num_frame_before_prediction - 1:]):
            state = variable.Variable(state)
            state_cost = mean_squared_error(state, gen_state) * 1e-4
            summaries.append(self.prefix + '_state_cost' + str(i) + ': ' + str(state_cost.data))
            loss += state_cost
        
        summaries.append(self.prefix + '_psnr_all: ' + str(psnr_all.data if isinstance(psnr_all, variable.Variable) else psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - num_frame_before_prediction)
        summaries.append(self.prefix + '_loss: ' + str(loss.data if isinstance(loss, variable.Variable) else loss))
        
        self.summaries = summaries
        self.gen_images = gen_images
        
        return self.loss


# =================================================
# Main entry point of the training processes (main)
# =================================================


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/brain-robotics-data/push/push_train', help='Directory containing data.')
@click.option('--output_dir', type=click.Path(), default='models', help='Directory for model checkpoints.')
@click.option('--event_log_dir', type=click.Path(), default='models', help='Directory for writing summary.')
@click.option('--epoch', type=click.INT, default=100000, help='Number of training iterations.')
@click.option('--pretrained_model', type=click.Path(), default='', help='Filepath of a pretrained model to initialize from.')
@click.option('--pretrained_state', type=click.Path(), default='', help='Filepath of a pretrained state to initialize from.')
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
@click.option('--validation_interval', type=click.INT, default=20, help='How often to run a batch through the validation model')
@click.option('--save_interval', type=click.INT, default=1, help='How often to save a model checkpoint')
@click.option('--debug', type=click.INT, default=0, help='Debug mode.')
def main(data_dir, output_dir, event_log_dir, epoch, pretrained_model, pretrained_state, sequence_length, context_frames, use_state, model_type, num_masks, schedsamp_k, train_val_split, batch_size, learning_rate, gpu, validation_interval, save_interval, debug):
    if debug == 1:
        chainer.set_debug(True)

    """ Train the model based on the data saved in ../processed """
    logger = logging.getLogger(__name__)
    logger.info('Training the model')

    logger.info('Model: {}'.format(model_type))
    logger.info('GPU: {}'.format(gpu))
    logger.info('# Minibatch-size: {}'.format(batch_size))
    logger.info('# epoch: {}'.format(epoch))

    model_suffix_dir = "{0}-{1}-{2}".format(time.strftime("%Y%m%d-%H%M%S"), model_type, batch_size)
    training_suffix = "{0}".format('training')
    validation_suffix = "{0}".format('validation')
    state_suffix = "{0}".format('state')

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
        images.append(np.float32(np.load(data_dir + '/' + data_map[i][2])))
        actions.append(np.float32(np.load(data_dir + '/' + data_map[i][3])))
        states.append(np.float32(np.load(data_dir + '/' + data_map[i][4])))

    images = np.asarray(images, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)

    train_val_split_index = int(np.floor(train_val_split * len(images)))
    images_training = np.asarray(images[:train_val_split_index])
    actions_training = np.asarray(actions[:train_val_split_index])
    states_training = np.asarray(states[:train_val_split_index])

    images_validation = np.asarray(images[train_val_split_index:])
    actions_validation = np.asarray(actions[train_val_split_index:])
    states_validation = np.asarray(states[train_val_split_index:])

    print('Data set contain {0}, {1} will be use for training and {2} will be use for validation'.format(len(images)-1, train_val_split_index, len(images)-1-train_val_split_index))

    # Create the model    
    training_model = Model(
        num_masks=num_masks,
        is_cdna=model_type == 'CDNA',
        is_dna=model_type == 'DNA',
        is_stp=model_type == 'STP',
        prefix='train'
    )
    validation_model = Model(
        num_masks=num_masks,
        is_cdna=model_type == 'CDNA',
        is_dna=model_type == 'DNA',
        is_stp=model_type == 'STP',
        prefix='val'
    )

    # Create the optimizers for the models
    optimizer = chainer.optimizers.Adam(alpha=learning_rate)
    optimizer.setup(training_model) 

    # Load a previous model
    if pretrained_model:
        chainer.serializers.load_npz(pretrained_model, model)
    if pretrained_state:
        chainer.serializers.load_npz(pretrained_state, optimizer)

    # Training
    # Enable GPU support if defined
    if gpu > -1:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    # Create the batches for Chainer's implementation of the iterator
    # Group the images, actions and states
    grouped_set_training = []
    grouped_set_validation = []
    for idx in xrange(len(images_training)):
        group = []
        group.append(images_training[idx])
        group.append(actions_training[idx])
        group.append(states_training[idx])
        grouped_set_training.append(group)
    for idx in xrange(len(images_validation)):
        group = []
        group.append(images_validation[idx])
        group.append(actions_validation[idx])
        group.append(states_validation[idx])
        grouped_set_validation.append(group)

    #train_iter = chainer.iterators.SerialIterator(grouped_set_training, batch_size)
    train_iter = chainer.iterators.SerialIterator(grouped_set_training, batch_size, repeat=True, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(grouped_set_validation, batch_size, repeat=False, shuffle=True)

    # Run training
    # As per Finn's implementation, one epoch is run on one batch size, randomly, but never more than once.
    # At the end of the queue, if the epochs len is not reach, the queue is generated again. 
    global_losses = []
    global_psnr_all = []
    global_losses_valid = []
    global_psnr_all_valid = []

    training_queue = []
    validation_queue = []
    fill_length_training = batch_size - (len(images_training) % batch_size)
    fill_length_validation = batch_size - (len(images_validation) % batch_size if len(images_validation) > batch_size else batch_size%len(images_validation))
    #for itr in xrange(epoch):
    while train_iter.epoch < epoch:
        itr = train_iter.epoch
        batch = train_iter.next()
        img_training_set, act_training_set, sta_training_set = concat_examples(batch)

        # Perform training
        logger.info("Begining training for mini-batch {0}/{1} of epoch {2}".format(str(train_iter.current_position), str(len(images_training)), str(itr+1)))
        #loss = training_model(img_training_set, act_training_set, sta_training_set, itr, schedsamp_k, use_state, num_masks, context_frames)
        optimizer.update(training_model, img_training_set, act_training_set, sta_training_set, itr, schedsamp_k, use_state, num_masks, context_frames)
        loss = training_model.loss
        psnr_all = training_model.psnr_all
        summaries = training_model.summaries

        global_losses.append(loss.data)
        global_psnr_all.append(psnr_all.data)

        logger.info("{0} {1}".format(str(itr+1), str(loss.data)))
        loss_valid, psnr_all_valid, summaries_valid = None, None, []

        if train_iter.is_new_epoch and itr+1 % validation_interval == 0:

            for batch in valid_iter:
                logger.info("Begining validation for mini-batch {0}/{1} of epoch {2}".format(str(valid_iter.current_position), str(len(images_validation)), str(itr+1)))
                img_validation_set, act_validation_set, sta_validation_set = concat_examples(batch)
                
                # Run through validation set
                #loss_valid, psnr_all_valid, summaries_valid = validation_model(img_validation_set, act_validation_set, sta_validation_set, itr, schedsamp_k, use_state, num_masks, context_frames)
                loss_valid = training_model(img_validation_set, act_validation_set, sta_validation_set, itr, schedsamp_k, use_state, num_masks, context_frames)
                psnr_all_valid = training_model.psnr_all
                summaries_valid = training_model.summaries

                global_losses_valid.append(loss_valid.data)
                global_psnr_all_valid.append(psnr_all_valid.data)
            
            valid_iter.reset()

        if train_iter.is_new_epoch and itr % save_interval == 0:
        #if itr % save_interval == 0:
            logger.info('Saving model')

            save_dir = output_dir + '/' + model_suffix_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            serializers.save_npz(save_dir + '/' + training_suffix + '-' + str(itr), training_model)
            #serializers.save_npz(save_dir + '/' + validation_suffix + '-' + str(itr), validation_model)
            serializers.save_npz(save_dir + '/' + state_suffix + '-' + str(itr), optimizer)
            np.save(save_dir + '/' + training_suffix + '-global_losses', np.array(global_losses))
            np.save(save_dir + '/' + training_suffix + '-global_psnr_all', np.array(global_psnr_all))
            np.save(save_dir + '/' + training_suffix + '-global_losses_valid', np.array(global_losses_valid))
            np.save(save_dir + '/' + training_suffix + '-global_psnr_all', np.array(global_psnr_all_valid))

        for summ in summaries:
            logger.info(summ)
        for summ_valid in summaries_valid:
            logger.info(summ_valid)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
