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
    se = tf.reduce_sum (tf.square(output - label))

    train_vars = tl.layers.get_variables_with_name('fully', True, True)
    opt = tf.train.AdamOptimizer(learningRate).minimize(
        loss, var_list=train_vars)

    return loss, se, opt


def train(features, labels, args):
    length = features.shape[0]
    train_length = int (args.validRatio * length + 0.5)
    valid_length = length - train_length

    train_features = features[:train_length, :]
    train_labels   = labels  [:train_length]
    valid_features = features[train_length:,:]
    valid_labels   = labels  [train_length:]

    print (features.shape, labels.shape)
    print ("train set:", train_features.shape, train_labels.shape)
    print ("valid set:", valid_features.shape, valid_labels.shape)

    sess = tf.InteractiveSession()

    # Build input pipepline training and validation data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))

    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(args.batchNum)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_features, valid_labels))
    valid_dataset = valid_dataset.batch(args.batchNum)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    feature, label = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)

    # Build model and optimizer
    with tf.variable_scope('train'):
        train_network = build_model(feature, args)

    with tf.variable_scope('valid'):
        args.mode = 'test'
        valid_network = build_model(feature, args)

    loss, se, opt = build_optimizer(train_network.outputs, label, args.learningRate)
    loss_v, se_v, opt_v = build_optimizer(valid_network.outputs, label, args.learningRate)

    initialize_global_variables(sess)

    if args.mode == 'retrain':
        tl.files.load_ckpt(sess, args.modelName, is_latest=False)

    #Training and Validation
    for epoch in range(args.epoch):

        #Training Stage
        total_train_se = 0
        debug = 0
        feed_dict = {}
        feed_dict.update(train_network.all_drop)
        sess.run(training_init_op)
        while True:
            try:
                se_, loss_, _, output_, label_ = sess.run([se, loss, opt, train_network.outputs, label], feed_dict=feed_dict)
                total_train_se = total_train_se + se_
                debug += len(output_)

            except tf.errors.OutOfRangeError:
                break

        print('Train data number: {}, {}'.format(debug, train_features.shape[0]))

        #Validation Stage
        train_variables_list = train_network.all_params
        valid_variables_list = valid_network.all_params
        for i in range(len(train_variables_list)):
            sess.run(tf.assign(valid_variables_list[i], train_variables_list[i]))

        if valid_features.shape[0] != 0:
            total_valid_se = 0
            debug = 0
            #dp_dict = tl.utils.dict_to_one( network.all_drop ) # disable noise layers
            #feed_dict = {}
            #feed_dict.update(dp_dict)

            sess.run(validation_init_op)
            while True:
                try:
                    se_ = sess.run(se_v, feed_dict=feed_dict)
                    total_valid_se = total_valid_se + se_
                    debug += len(output_)

                except tf.errors.OutOfRangeError:
                    break

        else:
            total_valid_se = -1 #Unknown validation loss

        print('Validation data number: {}, {}'.format(debug, valid_features.shape[0]))

        train_RMSE = np.sqrt (total_train_se/train_features.shape[0])
        valid_RMSE = np.sqrt (total_valid_se/valid_features.shape[0])
        print('Epoch: %d \t Train RMSE: %.4f, Validation RMSE: %.4f' % (epoch, train_RMSE, valid_RMSE))

    #TODO: tensorboard and evaluate validation error

    # Save Model
    tl.files.save_ckpt(sess, '%s' % (args.modelName), save_dir='checkpoint', var_list=train_network.all_params)

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
    if args.mode == 'train' or args.mode == 'retrain':
        features, labels = load_dataset(True)
        train (features, labels, args)

    elif args.mode == 'test':
        features, _ = load_dataset(False)
        evaluate (features, args)
