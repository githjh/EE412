import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def fully_connect_model(input_, layerNum, unitNum):
    with tf.variable_scope('fully'):
        network = InputLayer(input_)

        for i in range(layerNum):
            network = DenseLayer(network, unitNum, act=tf.nn.relu, W_init=tf.contrib.layers.variance_scaling_initializer(), name='dense%d'%(i))

        network = DenseLayer(network, 1, name='dense%d'%(layerNum))

    return network





