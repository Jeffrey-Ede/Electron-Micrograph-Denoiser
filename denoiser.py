import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config

import numpy as np
import cv2

import os

import functools
import itertools

import collections
import six

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

filters00 = 32
filters01 = 64
filters1 = 128
filters2 = 256
filters3 = 728
filters4 = 728
filters5 = 1024
filters6 = 1536
filters7 = 2048
numMiddleXception = 8

features0 = 64
features1 = 128
features2 = 256
features3 = 728
features4 = 728
aspp_filters = features4
aspp_output=256
aspp_size=32

aspp_rateSmall = 6
aspp_rateMedium = 12
aspp_rateLarge = 18

num_extra_blocks = 11

cropsize = 512
channels = 1

weight_decay = 5.e-5

def architecture(inputs, 
                 ground_truth, 
                 phase=False, 
                 params=None):
    """
    Atrous convolutional encoder-decoder noise-removing network
    phase - True during training
    """

    #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
    concat_axis = 3

    ##Reusable blocks
    def _batch_norm_fn(input):
        batch_norm = tf.contrib.layers.batch_norm(
            input,
            center=True, scale=True,
            is_training=False,
            fused=True,
            zero_debias_moving_mean=False,
            renorm=False)
        return batch_norm

    def batch_then_activ(input):
        batch_then_activ = _batch_norm_fn(input)
        batch_then_activ = tf.nn.relu6(batch_then_activ)
        return batch_then_activ

    def conv_block_not_sep(input, filters, kernel_size=3, phase=phase):
        """
        Convolution -> batch normalisation -> leaky relu
        phase defaults to true, meaning that the network is being trained
        """
        conv_block = slim.conv2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=kernel_size,
            padding="SAME",
            activation_fn=None)
        conv_block = batch_then_activ(conv_block)

        return conv_block

    def conv_block(input, filters, phase=phase):
        """
        Convolution -> batch normalisation -> leaky relu
        phase defaults to true, meaning that the network is being trained
        """
        conv_block = strided_conv_block(input, filters, 1, 1)

        return conv_block

    def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                           extra_batch_norm=True):
        
        strided_conv = slim.separable_convolution2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            depth_multiplier=1,
            stride=stride,
            padding='SAME',
            data_format='NHWC',
            rate=rate,
            activation_fn=None,#tf.nn.relu,
            normalizer_fn=_batch_norm_fn if extra_batch_norm else False,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=None,
            biases_initializer=tf.zeros_initializer(),
            biases_regularizer=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            scope=None)
        strided_conv = batch_then_activ(strided_conv)

        return strided_conv

    def deconv_block(input, filters, phase=phase):
        '''Transpositionally convolute a feature space to upsample it'''
        
        deconv_block = slim.conv2d_transpose(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            stride=2,
            padding="same",
            activation_fn=None)
        deconv_block = batch_then_activ(deconv_block)

        return deconv_block

    def aspp_block(input, phase=phase):
        """
        Atrous spatial pyramid pooling
        phase defaults to true, meaning that the network is being trained
        """

        ##Convolutions at multiple rates
        conv1x1 = slim.conv2d(inputs=input,
            num_outputs=aspp_filters,
            kernel_size=1,
            activation_fn=None,
            padding="same")
        conv1x1 = batch_then_activ(conv1x1)

        conv3x3_rateSmall = strided_conv_block(input=input,
                                     filters=aspp_filters,
                                     stride=1, 
                                     rate=aspp_rateSmall)
        conv3x3_rateSmall = batch_then_activ(conv3x3_rateSmall)

        conv3x3_rateMedium = strided_conv_block(input=input,
                                     filters=aspp_filters,
                                     stride=1, 
                                     rate=aspp_rateMedium)
        conv3x3_rateMedium = batch_then_activ(conv3x3_rateMedium)

        conv3x3_rateLarge = strided_conv_block(input=input,
                                     filters=aspp_filters,
                                     stride=1, 
                                     rate=aspp_rateLarge)
        conv3x3_rateLarge = batch_then_activ(conv3x3_rateLarge)

        #Image-level features
        pooling = tf.nn.pool(input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="SAME",
            strides=(2, 2))

        #Use 1x1 convolutions to project into a feature space the same size as
        #the atrous convolutions'
        #pooling = slim.conv2d(
        #    inputs=pooling,
        #    num_outputs=aspp_filters,
        #    kernel_size=1,
        #    activation_fn=None,
        #    padding="SAME")
        pooling = tf.image.resize_images(input, [aspp_size, aspp_size])
        pooling = batch_then_activ(pooling)

        #Concatenate the atrous and image-level pooling features
        concatenation = tf.concat(
            values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
            axis=concat_axis)

        #Reduce the number of channels
        reduced = slim.conv2d(
            inputs=concatenation,
            num_outputs=aspp_output,
            kernel_size=1,
            activation_fn=None,
            padding="SAME")
        reduced = batch_then_activ(reduced)

        return reduced

    def residual_conv(input, filters):

        residual = slim.conv2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=1,
            stride=2,
            padding="SAME",
            activation_fn=None)
        residual = batch_then_activ(residual)

        return residual

    def xception_middle_block(input, features):
        
        main_flow = strided_conv_block(
            input=input,
            filters=features,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=features,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=features,
            stride=1)

        return main_flow + input

    '''Model building'''
    input_layer = tf.reshape(inputs, [-1, cropsize, cropsize, channels])

    #Encoding block 0
    cnn0 = conv_block(
        input=input_layer, 
        filters=features0)
    cnn0_last = conv_block(
        input=cnn0, 
        filters=features0)
    cnn0_strided = strided_conv_block(
        input=cnn0_last,
        filters=features1,
        stride=2)

    residual0 = residual_conv(input_layer, features1)
    cnn0_strided += residual0

    #Encoding block 1
    cnn1 = conv_block(
        input=cnn0_strided, 
        filters=features1)
    cnn1_last = conv_block(
        input=cnn1, 
        filters=features1)
    cnn1_strided = strided_conv_block(
        input=cnn1_last,
        filters=features1,
        stride=2)

    residual1 = residual_conv(cnn0_strided, features1)
    cnn1_strided += residual1

    #Encoding block 2
    cnn2 = conv_block(
        input=cnn1_strided,
        filters=features2)
    cnn2_last = conv_block(
        input=cnn2,
        filters=features2)
    cnn2_strided = strided_conv_block(
        input=cnn2_last,
        filters=features2,
        stride=2)

    residual2 = residual_conv(cnn1_strided, features2)
    cnn2_strided += residual2

    #Encoding block 3
    cnn3 = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3_last = conv_block(
        input=cnn3,
        filters=features3)
    cnn3_strided = strided_conv_block(
        input=cnn3_last,
        filters=features3,
        stride=2)

    residual3 = residual_conv(cnn2_strided, features3)
    cnn3_strided += residual3

    #Encoding block 4
    cnn4 = conv_block(
        input=cnn3_strided,
        filters=features4)
    cnn4 = conv_block(
        input=cnn4,
        filters=features4)
    cnn4_last = conv_block(
        input=cnn4,
        filters=features4)

    cnn4_last += cnn3_strided

    for _ in range(num_extra_blocks):
        cnn4_last = xception_middle_block(cnn4_last, features4)

    ##Atrous spatial pyramid pooling
    aspp = aspp_block(cnn4_last)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    ##Decoding block 1 (deepest)
    #deconv4 = conv_block(aspp, features4)
    #deconv4 = conv_block(deconv4, features4)
    #deconv4 = conv_block(deconv4, features4)
    
    ##Decoding block 2
    #deconv4to3 = deconv_block(deconv4, features4)
    #concat3 = tf.concat(
    #    values=[deconv4to3, cnn3_last],
    #    axis=concat_axis)
    #deconv3 = conv_block(concat3, features3)
    #deconv3 = conv_block(deconv3, features3)
    #deconv3 = conv_block(deconv3, features3)

    deconv3 = tf.image.resize_images(aspp, [aspp_size*4, aspp_size*4])

    #Decoding block 3
    concat2 = tf.concat(
        values=[deconv3, cnn1_strided],
        axis=concat_axis)
    deconv2 = conv_block(concat2, features2)
    deconv2 = conv_block(deconv2, features2)

    residual2_d = conv_block_not_sep(concat2, features2, 1)
    deconv2 += residual2_d

    deconv2to1 = deconv_block(deconv2, features2)

    #Decoding block 4
    concat1 = tf.concat(
        values=[deconv2to1, cnn0_strided],
        axis=concat_axis)
    deconv1 = conv_block(concat1, features1)
    deconv1 = conv_block(deconv1, features1)
    
    residual1_d = conv_block_not_sep(concat1, features1, 1)
    deconv1 += residual1_d

    deconv1to0 = deconv_block(deconv1, features1)

    #Decoding block 5
    #concat0 = tf.concat(
    #    values=[deconv1to0, cnn0_last],
    #    axis=concat_axis)
    deconv0 = conv_block(deconv1to0, features0)
    deconv0 = conv_block(deconv0, features0)

    residual0_d = conv_block_not_sep(deconv1to0, features0, 1)
    deconv0 += residual0_d

    #Create final image with 1x1 convolutions
    deconv_final = conv_block_not_sep(deconv0, 1)

    #Image values will be between 0 and 1
    #output = tf.clip_by_value(
    #    deconv_final,
    #    clip_value_min=-0.1,
    #    clip_value_max=1.1,
    #    name='clipper')

    output = deconv_final

    return output


class RunConfig(tf.contrib.learn.RunConfig): 
    def uid(self, whitelist=None):
        """Generates a 'Unique Identifier' based on all internal fields.
        Caller should use the uid string to check `RunConfig` instance integrity
        in one session use, but should not rely on the implementation details, which
        is subject to change.
        Args:
          whitelist: A list of the string names of the properties uid should not
            include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
            includes most properties user allowes to change.
        Returns:
          A uid string.
        """
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        # Pop out the keys in whitelist.
        for k in whitelist:
            state.pop('_' + k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # For class instance without __repr__, some special cares are required.
        # Otherwise, the object address will be used.
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(), key=lambda t: t[0]))
        return ', '.join(
            '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))

def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the model."""

    def _model_fn(features, labels=None, mode=None, params=None):
        """Model body.

        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
        manages gradient updates.
    
        Args:
            features: a list of tensors, one for each tower
            mode: ModeKeys.TRAIN or EVAL
            params: Hyperparameters suitable for tuning
        Returns:
            An EstimatorSpec object.
        """
        is_training = mode#(mode == tf.estimator.ModeKeys.TRAIN)

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_grads = []
        tower_preds = []
        tower_mses = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = 'channels_last'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = local_device_setter(
                    worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('nn', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, grads, preds, mse = _tower_fn(
                            is_training, tower_features[i], tower_labels[i])

                        tower_losses.append(loss)
                        tower_grads.append(grads)
                        tower_preds.append(preds)
                        tower_mses.append(mse)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)


        _tower_losses_tmp = tf.tuple(tower_losses)
        _tower_mses_tmp = tf.tuple(tower_mses)
        _tower_preds_tmp = tf.stack(preds)
        
        return [_tower_losses_tmp, _tower_preds_tmp, _tower_mses_tmp, update_ops] + tower_grads

    return _model_fn


def _tower_fn(is_training, feature, ground_truth):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    #phase = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL
    output = architecture(feature[0], ground_truth[0], is_training)

    model_params = tf.trainable_variables()

    tower_pred = output

    out = tf.reshape(output, [-1, cropsize, cropsize, channels])
    truth = tf.reshape(ground_truth[0], [-1, cropsize, cropsize, channels])

    mse = tf.reduce_mean(tf.losses.mean_squared_error(out, truth))
    mse = tf.reshape(tf.cond(mse < 0.001, lambda: 1000.*mse, lambda: tf.sqrt(1000.*mse)), [1])
    tower_loss = mse

    tower_loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

    tower_loss = tf.reshape(tower_loss, (1,))

    tower_grad = tf.gradients(tower_loss, model_params)

    output_clipped = tf.clip_by_value(
        output,
        clip_value_min=0.,
        clip_value_max=1.,
        name='clipper-user_mse')
    out_clipped = tf.reshape(output_clipped, [-1, cropsize, cropsize, channels])
    mse_for_trainer = tf.reduce_mean(tf.losses.mean_squared_error(out_clipped, truth))

    return tower_loss, tower_grad, tower_pred, mse_for_trainer


class Denoiser(object):
    """Creates denoiser instance"""

    def __init__(self, 
                 checkpoint_loc="//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model",
                 visible_cuda=None):

        os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda

        # The env variable is on deprecation path, default is set to off.
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        #Session configuration.
        log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement,
            intra_op_parallelism_threads=0,
            gpu_options=tf.GPUOptions(force_gpu_compatible=True))

        config = RunConfig(
            session_config=sess_config, model_dir=checkpoint_loc)

        temp = set(tf.all_variables())
        sess = tf.Session(config=sess_config)
        sess.run(tf.initialize_variables(set(tf.all_variables())-temp))
        temp = set(tf.all_variables())

        img_ph = [tf.placeholder(tf.float32, shape=(1,512,512,1), name='img')]
        img_truth_ph = [tf.placeholder(tf.float32, shape=(1,512,512,1), name='img_truth')]

        model_fn = get_model_fn(num_gpus=1, variable_strategy='GPU', num_workers=1)

        is_training = False
        hparams=None
        results = model_fn(img_ph, img_truth_ph, mode=is_training, params=hparams)
        self._tower_preds = results[1]

        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        temp = set(tf.all_variables())

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_loc))

        self.sess = sess
        self.img_ph = img_ph

    def preprocess(self, img):

        img[np.isnan(img)] = 0.5
        img[np.isinf(img)] = 0.5

        img = scale0to1(img).reshape(1,512,512,1)

        return img

    def denoise_crop(self, crop, preprocess=True, scaling=True, postprocess=True):

        crop = cv2.resize(crop, (512,512))

        if scaling:
            offset = np.min(crop)
            scale = np.max(crop) - offset
            
            if scale:
                crop = (crop-offset)/scale
            else: 
                crop = crop.fill(0.5)

        pred = self.sess.run(self._tower_preds, 
                             feed_dict={self.img_ph[0]: 
                                        self.preprocess(crop) if preprocess else crop})

        if scaling:
            pred = pred*scale+offset if scale else pred*offset/np.mean(pred)

        if postprocess:
            return (crop).clip(0., 1.).reshape(512, 512)
        else:
            return pred

    def denoise(img, preprocess=True, postprocess=True, overlap=10, half_size=True):
        """
        img: Image to denoise
        preprocess: Remove nans and infs
        """
        
        if half_size:
            img = cv2.resize(img, (img.size[0]//2,img.size[1]//2))

        if preprocess:
            img = self.preprocess(img)

        denoised = np.zeros(img.shape)
        contributions = np.zeros(img.shape)
    
        num0 = img.shape[0]//(512-overlap)+1
        num1 = img.shape[1]//(512-overlap)+1
        len0 = img.shape[0]/num0
        len1 = img.shape[1]/num1

        for i in range(num0):
            x = np.round(i*len0)
            for j in range(num1):
                y = np.round(j*len1)

                crop = img[x:(x+512), y:(y+512)]
                offset = np.min(crop)
                scale = np.max(crop) - offset

                if scale:
                    crop = (crop-offset)/scale
                else: 
                    crop = crop.fill(0.5)

                pred = self.denoise_crop(
                    crop=crop, 
                    preprocess=False,
                    scaling=False, 
                    postprocess=False).reshape((512,512))

                pred = pred*scale+offset if scale else pred*offset/np.mean(pred)

                denoised[x:(x+512), y:(y+512)] = pred*scale + offset
                contributions[x:(x+512), y:(y+512)] += 1

        denoised /= contributions

        if postprocess:
            return denoised.clip(0., 1.)
        else:
            return denoised

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img = img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)

    return

if __name__ == "__main__":

    denoiser = Denoiser(visible_cuda="0")
    disp(denoiser.denoise_crop(np.random.rand(512,512)))
