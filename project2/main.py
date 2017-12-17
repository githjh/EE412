import csv
import sys
import argparse
import tensorflow as tf
import tensorlayer as tl
from dataset import *
from model import *
from option import *


def build_model(feature, args):
    model = 'None'

    if args.model == 'fully':
        model = fully_connect_model(
            feature, args.layerNum, args.unitNum, args.activator, args.initializer, args.model)

    elif args.model == 'fully_BN':
        if args.mode == 'train' or args.mode == 'retrain':
            model = fully_connect_BN_model(
                feature, args.layerNum, args.unitNum, args.activator, args.initializer, True, args.decay, args.model)
        else:
            model = fully_connect_BN_model(
                feature, args.layerNum, args.unitNum, args.activator, args.initializer, False, args.decay, args.model)

    elif args.model == 'fully_dropout':
        if args.mode == 'train' or args.mode == 'retrain':
            model = fully_connect_dropout_model(
                feature, args.layerNum, args.unitNum, args.activator, args.initializer, args.keep, True, args.model)
        else:
            model = fully_connect_dropout_model(
                feature, args.layerNum, args.unitNum, args.activator, args.initializer, 1, False, args.model)

    assert model is not None

    return model


def build_optimizer(output, label, learningRate):
    diff = output - label
    #loss = tl.cost.mean_squared_error(output, label, is_mean=True)
    loss = tf.reduce_mean (tf.square(output - label))
    sum_se = tf.reduce_sum (tf.square(output - label))

    train_vars = tl.layers.get_variables_with_name('fully', True, True)
    opt = tf.train.AdamOptimizer(learningRate).minimize(
        loss, var_list=train_vars)

    return sum_se, loss, opt


def train(features, labels, args):
    sess = tf.InteractiveSession()
    print ("train")

    # Build input pipepline
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(args.batchNum)
    iterator = dataset.make_initializable_iterator()

    feature, label = iterator.get_next()

    # Build model and optimizer
    network = build_model(feature, args)
    sum_se, loss, opt = build_optimizer(network.outputs, label, args.learningRate)

    initialize_global_variables(sess)

    if args.mode == 'retrain':
        tl.files.load_ckpt(sess, args.modelName, is_latest=False)
        sys.exit (-1)

    ### tensorlayer already handle error ###
    """
    if args.mode == 'retrain':
        try:
            tl.files.load_ckpt(sess, args.modelName, is_latest=False, var_list=network.all_params)
        except tf.errors.NotFoundError:
            logging.exception('fail to load checkpoint file')
            sys.exit()
    """

    #print(tl.layers.get_variables_with_name(args.model))
    feed_dict = {}
    feed_dict.update(network.all_drop)

    # Train
    for epoch in range(args.epoch):
        total_loss = 0
        total_iter = 0
        total_se = 0
        sess.run(iterator.initializer)

        while True:
            try:
                sum_se_, loss_, _, output_, label_ = sess.run([sum_se, loss, opt, network.outputs, label], feed_dict=feed_dict)
                total_se += sum_se_
                total_loss = total_loss + loss_
                total_iter = total_iter + 1  # might be subtituded by global step
            except tf.errors.OutOfRangeError:
                break

        #TODO: tensorboard and evaluate validation error
        rmse = np.sqrt (total_se/features.shape[0])
        print('Epoch: %d \t Average Train Error: %.4f, rmse: %.4f' %
              (epoch, total_loss / total_iter, rmse))

    # Save Model
    tl.files.save_ckpt(sess, '%s' % (args.modelName), save_dir='checkpoint', var_list=network.all_params)

def valid (features, labels, args):
    # Load Model
    sess = tf.InteractiveSession()
    print ("valid")

    # Build input pipepline
    #dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    batched_dataset = dataset.batch(args.batchNum)
    iterator = batched_dataset.make_one_shot_iterator()

    #feature = iterator.get_next()
    feature, label = iterator.get_next()

    # Build model and optimizer
    network = build_model(feature, args)

    initialize_global_variables(sess)
    tl.files.load_ckpt(sess, args.modelName, is_latest=False)
    feed_dict = {}
    feed_dict.update(network.all_drop)
    sum_se, loss, _ = build_optimizer(network.outputs, label, args.learningRate)

    sum_se_, loss_, output_, label_ = sess.run([sum_se, loss, network.outputs, label], feed_dict=feed_dict)

    rmse = np.sqrt (loss_)
    print ("validation: loss: %.4f, rmse: %.4f" % (loss_, rmse))

# TODO: quantize output value [-5, 5]

def evaluate(features, args):
    # Load Model
    sess = tf.InteractiveSession()
    print ("test")

    # Build input pipepline
    dataset = tf.data.Dataset.from_tensor_slices(features)
    batched_dataset = dataset.batch(args.batchNum)
    iterator = batched_dataset.make_one_shot_iterator()

    feature = iterator.get_next()

    # Build model and optimizer
    network = build_model(feature, args)

    initialize_global_variables(sess)
    tl.files.load_ckpt(sess, args.modelName, is_latest=False)
    sys.exit (-1)

    # Evaluate
    output_file_name = 'result.txt'
    with open(output_file_name, 'w') as fo:
        data_writer = csv.writer(fo)
        while True:
            try:
                feature_, output_ = sess.run([feature, network.outputs])
                for i in range(feature_.shape[0]):
                    user_id = np.argmax(feature_[i][0:USER_MAX]) + 1
                    item_id = np.argmax(
                        feature_[i][USER_MAX:USER_MAX + ITEM_MAX]) + 1
                    data_writer.writerow([user_id, item_id, output_[i][0]])
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    
    if args.mode == 'test':
        features, _ = load_dataset(False)
        evaluate (features, args)

    else:
        features, labels = load_dataset(True)

        length = features.shape[0]
        train_length = int (0.8 * length + 0.5)
        valid_length = length - train_length

        train_features = features[:train_length, :]
        train_labels   = labels  [:train_length]
        valid_features = features[train_length:,:]
        valid_labels   = labels  [train_length:]

        print (features.shape, labels.shape)
        print ("train set:", train_features.shape, train_labels.shape)
        print ("valid set:", valid_features.shape, valid_labels.shape)

        if args.mode == 'train' or args.mode == 'retrain':
            train (train_features, train_labels, args)

        elif args.mode == "valid":
            valid (valid_features, valid_labels, args)

