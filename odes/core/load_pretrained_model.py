"""
Description: loading the pretrained model.

    The pretrained models can include models converted from Caffemodel, saved in **.npy, or the checkpoint in tensorflow format.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def import_pretrained_model_from_npy():
    """
    Descriptions:

    Args:

    Returns:

    """
    #assert ## checking format of input args

    raise NotImplementedError


def import_pretrained_models_from_ckpt( sess, pretrained_model):
    """
    Description:(one branch per sentence)

    Args:

    Returns:

    """
    #assert ## checking format of input args
    print('Loading initial model weights from {:s}'.format(pretrained_model))
    variables = tf.global_variables()
    sess.run(tf.variables_initializer(variables, name='init'))
    # getting all variables in pretrained model
    var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)
    # selecting restored variables
    variables_to_restore = get_variables_to_restore(variables, var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    ## laoding the pre-trained model
    restorer.restore(sess, pretrained_model)


    #raise NotImplementedError


def get_variables_to_restore(variables, var_keep_dic):
    "getting variable to restore"
    ## CAUTION: some input weight are BRG instead of RGB,
    variables_to_restore = []
    for v in variables:
        if v.name.split(':')[0] == "global_step":  ## this one different data format
            continue
        if v.name.split(':')[0] in var_keep_dic:
            print("variable to restore: {}".format(v))
            variables_to_restore.append(v)

    return variables_to_restore


def get_variables_in_checkpoint_file(pretrained_model):
    ##one sentence one branch, checking everytime
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY" )
