import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from option import *

def fully_connect_model(input_, layerNum, unitNum, act, W_init, scope_name):
    with tf.variable_scope(scope_name):
        network = InputLayer(input_)

        for i in range(layerNum):
            network = DenseLayer(network, unitNum, act=function_dict[act], W_init=function_dict[W_init](**param_dict[W_init]), name='dense%d'%(i))

        network = DenseLayer(network, 1, name='dense%d'%(layerNum))

    return network

def fully_connect_BN_model(input_, layerNum, unitNum, act, W_init, is_train, decay, scope_name):
    with tf.variable_scope(scope_name):
        network = InputLayer(input_)

        for i in range(layerNum):
            network = DenseLayer(network, unitNum, act=function_dict[act], W_init=function_dict[W_init](**param_dict[W_init]), name='dense%d'%(i))
            network = BatchNormLayer(network, act=function_dict[act], is_train=is_train, decay=decay, name='batchnorm%d'%(i))

        network = DenseLayer(network, 1, name='dense%d'%(layerNum))

    return network

def fully_connect_dropout_model(input_, layerNum, unitNum, act, W_init, keep, is_train, scope_name):
    with tf.variable_scope(scope_name):
        network = InputLayer(input_)

        for i in range(layerNum):
            network = DropoutLayer(network, keep=keep, is_train=is_train, name='dropout%d'%(i))
            network = DenseLayer(network, unitNum, act=function_dict[act], W_init=function_dict[W_init](**param_dict[W_init]), name='dense%d'%(i))

        network = DenseLayer(network, 1, name='dense%d'%(layerNum))

    return network

