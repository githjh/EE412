import argparse
import tensorflow as tf
import tensorlayer as tl
from dataset import *
from model import *

def build_optimizer(output, label, learningRate):
    diff = output - label
    #loss = tl.cost.mean_squared_error(output, label, is_mean=True)
    loss = tf.reduce_mean(tf.square(output - label))

    train_vars = tl.layers.get_variables_with_name('fully', True, True)
    opt = tf.train.AdamOptimizer(learningRate).minimize(loss, var_list=train_vars)

    return loss, opt


def train(features, labels, args):
    sess = tf.InteractiveSession()

    #Build input pipepline
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(args.batchNum)
    #datset = dataset.repeat(3)  # Repeat the input indefinitely.
    iterator = dataset.make_initializable_iterator()

    feature, label = iterator.get_next()

    #Build model and optimizer
    network = fully_connect_model(feature, args.layerNum, args.unitNum)
    loss, opt = build_optimizer(network.outputs, label, args.learningRate)

    initialize_global_variables(sess)

    #Train
    for epoch in range(args.epoch):
        total_loss = 0
        total_iter = 0
        sess.run(iterator.initializer)

        while True:
            try:
                loss_, _ = sess.run([loss, opt])
                total_loss = total_loss + loss_
                total_iter = total_iter + 1 # might be subtituded by global step
            except tf.errors.OutOfRangeError:
                break

        print('Epoch: %d \t Average Train Error: %.4f' % (epoch, total_loss / total_iter))

    #Save Model
    tl.files.save_ckpt(sess, 'project2.ckpt', save_dir='checkpoint', var_list=tl.layers.get_variables_with_name('fully'))

def evaluate(features, args):
    #Load Model
    sess = tf.InteractiveSession()

    #Build input pipepline
    dataset = tf.data.Dataset.from_tensor_slices((features))
    #dataset = dataset.shuffle(buffer_size=10000)
    batched_dataset = dataset.batch(args.batchNum)
    iterator = batched_dataset.make_one_shot_iterator()

    feature = iterator.get_next()

    #Build model and optimizer
    network = fully_connect_model(feature, args.layerNum, args.unitNum)
    tl.files.load_ckpt(sess, args.modeName)

    initialize_global_variables(sess)

    total_loss = 0; total_iter = 0;
    #Evaluate
    while True:
        try:
            loss_, _ = sess.run([loss, opt, network.outputs])
            total_loss = total_loss + loss_
            total_iter = total_iter + 1 # might be subtituded by global step
        except tf.errors.OutOfRangeError:
            break

    print('Average Test Error: %.4f' % (total_loss / total_iter))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layerNum', default=2, type=int)
    parser.add_argument('--unitNum', default=100, type=int)
    parser.add_argument('--batchNum', default=64, type=int)
    parser.add_argument('--learningRate', default=1e-4, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--modelName', default='', type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'])

    args = parser.parse_args()
    if args.mode is 'train':
        features, labels = load_dataset(True)
        train(features, labels, args)
    else: #TODO: test evaluation stage
        features = load_dataset(False)
        evaluate(features, labels, args)
