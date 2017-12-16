import csv
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

    if args.mode == 'retrain':
        tl.files.load_ckpt(sess, args.modelName, is_latest=False)

    ### tensorlayer already handle error ###
    """
    if args.mode == 'retrain':
        try:
            tl.files.load_ckpt(sess, args.modelName, is_latest=False, var_list=network.all_params)
        except tf.errors.NotFoundError:
            logging.exception('fail to load checkpoint file')
            sys.exit()
    """

    #Train
    for epoch in range(args.epoch):
        total_loss = 0
        total_iter = 0
        sess.run(iterator.initializer)

        while True:
            try:
                loss_, _, output_, label_ = sess.run([loss, opt, network.outputs, label])
                total_loss = total_loss + loss_
                total_iter = total_iter + 1 # might be subtituded by global step
            except tf.errors.OutOfRangeError:
                break

        print('Epoch: %d \t Average Train Error: %.4f' % (epoch, total_loss / total_iter))
        #print(output_)
        #print(label_)

    #Save Model
    tl.files.save_ckpt(sess, '%s'%(args.modelName), save_dir='checkpoint', var_list=tl.layers.get_variables_with_name('fully'))

#TODO: quantize output value [-5, 5]
def evaluate(features, args):
    #Load Model
    sess = tf.InteractiveSession()

    #Build input pipepline
    dataset = tf.data.Dataset.from_tensor_slices(features)
    #dataset = dataset.shuffle(buffer_size=10000)
    batched_dataset = dataset.batch(args.batchNum)
    iterator = batched_dataset.make_one_shot_iterator()

    feature = iterator.get_next()

    #Build model and optimizer
    network = fully_connect_model(feature, args.layerNum, args.unitNum)

    initialize_global_variables(sess)
    tl.files.load_ckpt(sess, args.modelName, is_latest=False)

    #Evaluate
    output_file_name = 'result.txt'
    with open(output_file_name,'w') as fo:
        data_writer = csv.writer(fo)
        while True:
            try:
                feature_, output_ = sess.run([feature, network.outputs])
                for i in range(feature_.shape[0]):
                    user_id = np.argmax(feature_[i][0:USER_MAX]) +1
                    item_id = np.argmax(feature_[i][USER_MAX:USER_MAX+ITEM_MAX]) +1
                    data_writer.writerow([user_id,item_id,output_[i][0]])
            except tf.errors.OutOfRangeError:
                break

    #print('Average Test Error: %.4f' % (total_loss / total_iter))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layerNum', default=2, type=int)
    parser.add_argument('--unitNum', default=100, type=int)
    parser.add_argument('--batchNum', default=64, type=int)
    parser.add_argument('--learningRate', default=1e-4, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--modelName', type=str, required=True)
    parser.add_argument('--mode', default='train', choices=['train', 'retrain', 'test'])

    args = parser.parse_args()

    print(args.modelName)
    if args.mode == 'train' or args.mode == 'retrain':
        features, labels = load_dataset(True)
        print(features.shape, labels.shape)
        train(features, labels, args)
    else: #TODO: test evaluation stage
        features, _ = load_dataset(False)
        #features = np.array([
        evaluate(features, args)
