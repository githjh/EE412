import argparse
import tensorflow as tf
import tensorlayer as tl

parser = argparse.ArgumentParser()

#network architecture
parser.add_argument('--model', default='fully', choices=['fully', 'fully_BN', 'fully_dropout'])
parser.add_argument('--layerNum', default=2, type=int)
parser.add_argument('--unitNum', default=100, type=int)

#network hyper-parameter
parser.add_argument('--activator', default='relu', choices=['tanh', 'relu', 'lrelu', 'prelu']) #TODO: prelu
parser.add_argument('--initializer', default='he', choices=['truncated_normal', 'xavier', 'he'])

#layer hyper-parameter (ex. lrelu, batch normalization, dropout, ...)
parser.add_argument('--lrelu_slope', default=0.1, type=float)
parser.add_argument('--normal_stddev', default=0.1, type=float)
parser.add_argument('--keep', default=0.5, type=float, help='dropout keep probability')
parser.add_argument('--batchnorm_decay', default=0.9, type=float, help='batch normalization decay')
parser.add_argument('--l2_decay', default=1.25e-3, type=float, help='batch normalization decay')

#training hyper-parameter
parser.add_argument('--batchNum', default=64, type=int)
parser.add_argument('--learningRate', default=1e-4, type=float)
parser.add_argument('--epoch', default=50, type=int)

#others
parser.add_argument('--modelName', type=str, required=True)
parser.add_argument('--mode', default='train', choices=['train', 'retrain', 'test'])
parser.add_argument('--validRatio', default=0.8, type=float)

args = parser.parse_args()

function_dict= {
    'truncated_normal': tf.truncated_normal_initializer,
    'xavier': tf.contrib.layers.xavier_initializer,
    'he': tf.contrib.layers.variance_scaling_initializer,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'lrelu': lambda x : tl.act.lrelu(x, args.lrelu_slope)
    }

param_dict = {
    'truncated_normal': {'stddev': args.normal_stddev},
    'xavier': {},
    'he': {},
    }
