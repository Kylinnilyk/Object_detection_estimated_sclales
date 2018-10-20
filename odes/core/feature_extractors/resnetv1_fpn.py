import tensorflow as tf

from odes.core.feature_extractors import img_feature_extractor


import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block


#resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim

#fixed_block = 1  ## benz, you need to put this one into pyramid_config file


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer( weight_decay ),
        weights_initializer=slim.variance_scaling_initializer(),
        trainable=is_training,
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class resnetv1_fpn(img_feature_extractor.ImgFeatureExtractor):
    """Modified Resnet 101 model definition to extract features from
    RGB image input using pyramid features.
    """
    def _build_base(self, input_img):
        with tf.variable_scope(self._scope, self._scope):
            net = resnet_utils.conv2d_same( input_img, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net



    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='resnet_v1_101',
              rpn_weight_decay=0.0001 ):  ## scope is important variable to set
        """ resnet
        args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        res_config = self.config
        fixed_block = res_config.fixed_block
        self._scope = scope
        rpn_weight_decay = res_config.rpn_weight_decay

        ## backbone
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_base = self._build_base(inputs)


        blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                  resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                  resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),   ##decreasing factor is 32
                  resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]



        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv1, net_dict1 = resnet_v1.resnet_v1(net_base,
                                              blocks[0:fixed_block],
                                              global_pool=False,
                                              include_root_block=False,  ## no resue
                                              scope=self._scope)


        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv2, net_dict2 = resnet_v1.resnet_v1(net_conv1,
                                              blocks[fixed_block:],
                                              global_pool=False,
                                              include_root_block=False,
                                              scope=self._scope)

        ## build feature maps
        feature_maps_dict  = {
            'C2': net_dict1['resnet_v1_101/block1/unit_2/bottleneck_v1'],
            'C3': net_dict2['resnet_v1_101/block2/unit_3/bottleneck_v1'],
            'C4': net_dict2['resnet_v1_101/block3/unit_22/bottleneck_v1'],
            'C5': net_dict2['resnet_v1_101/block4']
        }


        ## build pyramid feature maps
        feature_pyramid = {}

        upsample_method = 'deconv'   ## put this setting into configuration file later

        if upsample_method == 'deconv':
            ## using deconvolution to build pyramid feature image
            with tf.variable_scope('build_feature_pyramid'):
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer( rpn_weight_decay )):
                     feature_pyramid['P5'] = slim.conv2d(feature_maps_dict['C5'],
                                                        num_outputs=256,
                                                        kernel_size=[1, 1],
                                                        stride=1,
                                                        scope='build_P5')
                     ##p6 is down sample of p5
                     for layer in range(4, 1, -1):
                         p, c = feature_pyramid['P' + str(layer + 1)], feature_maps_dict['C' + str(layer)]

                         up_sample = slim.conv2d_transpose(
                                         p,
                                         num_outputs=256 ,
                                         kernel_size=[3, 3],
                                         stride=2,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={
                                             'is_training': is_training},
                                         scope ='build_P%d/up_sampling_deconvolution' % layer)

                         c = slim.conv2d(c,
                                         num_outputs=256,
                                         kernel_size=[1, 1],
                                         stride=1,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={
                                            'is_training': is_training },
                                         scope='build_P%d/reduce_dimension' % layer)

                         concat = tf.concat(
                             (c, up_sample), axis=3, name='build_P%d/concate_layer' %layer)

                         p = slim.conv2d(concat,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={
                                             'is_training': is_training},
                                         scope= 'build_P%d/for_feature_pyramid' % layer)

                         feature_pyramid['P' + str(layer)] = p



        else:
            ## building pyramid with summation instead of deconvolution operation,
            with tf.variable_scope('build_feature_pyramid'):
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer( rpn_weight_decay )):
                    feature_pyramid['P5'] = slim.conv2d(feature_maps_dict['C5'],
                                                        num_outputs=256,
                                                        kernel_size=[1, 1],
                                                        stride=1,
                                                        scope='build_P5')
                    #feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                    #                                        kernel_size=[2, 2], stride=2, scope='build_P6')
                    ##p6 is down sample of p5
                    for layer in range(4, 1, -1):
                        p, c = feature_pyramid['P' + str(layer + 1)], feature_maps_dict['C' + str(layer)]
                        up_sample_shape = tf.shape(c)
                        up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                                     name='build_P%d/up_sample_nearest_neighbor' % layer)
                        c = slim.conv2d(c,
                                        num_outputs=256,
                                        kernel_size=[1, 1],
                                        stride=1,
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                             'is_training': is_training},
                                        scope='build_P%d/reduce_dimension' % layer)
                        p = up_sample + c
                        p = slim.conv2d(p,
                                        num_outputs=256,
                                        kernel_size=[3, 3],
                                        stride=1,
                                        padding='SAME',
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                             'is_training': is_training},
                                        scope='build_P%d/avoid_aliasing' % layer)
                        feature_pyramid['P' + str(layer)] = p




        feature_maps_out = feature_pyramid['P2']

        return feature_maps_out, feature_maps_dict  #end_points


    ## this one is like the google object detection model manner
    def build_2(self,
                inputs,
                input_pixel_size,
                is_training,
                scope='resnet_v1_101',
                weight_decay=0.0001
                ):
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         num_classes = None,
                                                         is_training= is_training,
                                                         global_pool= False,
                                                         output_stride= None,
                                                         spatial_squeeze= False)

        feature_maps_dict  = {
            'C2': self.share_net['resnet_v1_101/block1/unit_2/bottleneck_v1'],
            'C3': self.share_net['resnet_v1_101/block2/unit_3/bottleneck_v1'],
            'C4': self.share_net['resnet_v1_101/block3/unit_22/bottleneck_v1'],
            'C5': self.share_net['resnet_v1_101/block4']
        }
        feature_maps_out = feature_maps_dict['C5']

        return feature_maps_out, feature_maps_dict
