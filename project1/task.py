# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:22:47 2017

@author: user
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time
import argparse
import os, sys
import copy

# Read training data from csv train file
train_headers = ["suit1","rank1","suit2","rank2","suit3","rank3","suit4","rank4","suit5","rank5","hand"]
data_test_headers = ["suit1","rank1","suit2","rank2","suit3","rank3","suit4","rank4","suit5","rank5"]
hand_test_headers = ["hand"]

train_dtype_dict = {"suit1":int,"rank1":int,"suit2":int,"rank2":int,"suit3":int,"rank3":int,"suit4":int,"rank4":int,"suit5":int,"rank5":int,"hand":int}
data_test_dtype_dict = {"suit1":int,"rank1":int,"suit2":int,"rank2":int,"suit3":int,"rank3":int,"suit4":int,"rank4":int,"suit5":int,"rank5":int}
hand_test_dtype_dict = {"hand":int}

train_file = "./train_data.csv"
test_data_file = "./test_data.csv"
test_label_file = "./test_hand.csv"

class Card (object):
    def __init__ (self, suit, rank):
        self.suit = suit
        self.rank = rank

    def modify_card (self, suit2, rank2):
        self.suit = suit2
        self.rank = rank2

    #@multimethod(int,int)
    """def is_same_card (self, suit2, rank2):
        return self.suit == suit2 and self.rank1 == rank2"""

    #@multimethod(object)
    def is_same_card (self, card2):
        return self.suit == card2.suit and self.rank == card2.rank

    def get_card (self):
        list_suit_rank = [self.suit, self.rank]
        return list_suit_rank

    def __str__ (self, pre_str = "") :
        return pre_str + str (self.get_card())


class GroupCard (Card):

    def __init__ (self, list_cards, hand=-1):
        """
            - Args:
                + list_cards: [card1, card2, .., card5]
                 with cardi = [cardi.suit, cardi.rank]
        """
        self.size_group = 5
        self.list_cards = list_cards
        self.hand = hand

    def copy (self):
        return GroupCard (copy.copy (self.list_cards), copy.copy (self.hand))

    def modify_a_card (self, new_card, index):
        self.list_cards[index] = new_card

    def change_hand (self, hand):
        self.hand = hand

    def get_group_card (self):
        list_suit_rank = []
        for i in range (self.size_group):
            list_suit_rank += self.list_cards[i].get_card()
        return list_suit_rank

    def zeros (self):
        card1 = Card (0, 0)
        card2 = Card (0, 0)
        card3 = Card (0, 0)
        card4 = Card (0, 0)
        card5 = Card (0, 0)

        list_card = [card1, card2, card3, card4, card5]

        groupCard = GroupCard (list_card)
        return groupCard

    def get_group_card_arr (self):
        return np.array (self.get_group_card()).reshape (1, 2 * self.size_group)

    def __str__ (self, pre_str = "") :
        return pre_str + str (self.get_group_card_arr()) + "," + str (self.hand)

class Task1(GroupCard):

    def __init__(self, train_file, test_data_file, test_label_file, args):
        #from args
        self.gpu_idx = args.gpu_idx
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.hidden_layer_num = args.hidden_layer_num
        self.neuron_num = args.neuron_num

        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        self.decay_rate = args.decay_rate
        self.saved_period = args.saved_period
        self.mode = args.mode

        self.model_dir = args.model_dir
        self.model_name = args.model_name

        self.train_file = train_file
        self.test_data_file = test_data_file
        self.test_label_file = test_label_file

        self.train_data = self.get_data (self.train_file, train_headers, train_dtype_dict)[0]
        self.train_hand = self.get_data (self.train_file, train_headers, train_dtype_dict)[1]
        self.test_data = self.get_data (self.test_data_file, data_test_headers, data_test_dtype_dict)
        self.test_hand = self.get_data (self.test_label_file, hand_test_headers, hand_test_dtype_dict)
        self.enc = OneHotEncoder()
        self.enc.fit(np.array ([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]))

        self.args = args
        self.dim_hand = args.dim_hand#self.train_hand.shape[1]
        self.dim_data = args.dim_data#self.train_data.shape[1]

        self.DROP_OUT = 1#0.75


    def get_data (self, file_name, headers, dtype):
        """ get data from train_data.csv, or test_data.csv, or test_hand.csv.
        """
        train_data = pd.read_csv (file_name, names = headers, dtype = dtype, header = -1)
        Z = np.array ([train_data[headers[0]]]).T
        no_column = len (headers)

        if (no_column > 1):

            if  file_name == self.train_file:
                no_features = no_column - 1
            else:
                no_features = no_column

            for i in range (1, no_features):
                column_i = np.array ([train_data[headers[i]]]).T
                Z = np.concatenate ( (Z, column_i), axis = 1)

            if  file_name == self.train_file:
                y = np.array ([train_data[headers[no_features]]]).T
                return (Z, y)

        return Z

        #create model
    def build_model (self,dim_data, dim_hand, no_unit_in_a_hidden_layer, layer_num, no_epoch, batch_size):
        weights = []

        X = tf.placeholder(tf.float32, [None, dim_data])
        Y = tf.placeholder(tf.float32, [None, dim_hand])

        #dropout = tf.placeholder(tf.float32, name='dropout')

        W1 = tf.Variable(tf.random_normal([dim_data, no_unit_in_a_hidden_layer], stddev=0.01))
        B1 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer], stddev=0.01))
        L1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W1), B1))

        W2 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, no_unit_in_a_hidden_layer], stddev=0.01))
        B2 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer], stddev=0.01))
        L2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(L1, W2), B2))
        #L2_hat = tf.nn.dropout(L2, dropout, name='relu_dropout')

        W3 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, dim_hand], stddev=0.01))
        B3 = tf.Variable(tf.random_normal([dim_hand], stddev=0.01))
        logits = tf.nn.bias_add(tf.matmul(L2, W3), B3)

        weights.append(W1)
        weights.append(W2)
        weights.append(W3)
        weights.append(B1)
        weights.append(B2)
        weights.append(B3)

        return X, Y, logits, weights

    def train_nn(self):

        X, Y, logits, weights = self.build_model(self.dim_data, self.dim_hand, self.neuron_num, self.hidden_layer_num, self.epoch, self.batch_size)

        num_batches_per_epoch = int(len(self.train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        #decay_steps = num_batches_per_epoch
        global_step = tf.Variable(0, trainable = False)
        learning_rate = 0.001#tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

        """ Initialize the variables with default values"""
        init = tf.global_variables_initializer()
        #saver = tf.train.Saver(weights, max_to_keep=0)

        with tf.Session() as sess:
            saver = tf.train.Saver()

            # Run the initializer
            sess.run(init)

            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()

            train_data = self.train_data
            train_hand = self.train_hand
            train_set = np.concatenate ((train_data, train_hand), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_hand_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            total_batch = int(len(self.train_data)/self.batch_size)

            for epoch in range (self.epoch):

                total_cost = 0
                dropoutotal_cost = 0
                index_counter = 0

                for i in range (total_batch):
                    start_index = index_counter * self.batch_size
                    end_index = (index_counter + 1) * self.batch_size
                    batch_x = train_data_shuffled [start_index : end_index]
                    batch_y = train_hand_shuffled [start_index : end_index]
                    batch_y_encode = self.enc.transform (batch_y).toarray() #batch_y#
                    #print ("batch_x", batch_x)
                    #print ("batch_y", batch_y)
                    index_counter = index_counter + 1

                    if (index_counter >= total_batch):
                        index_counter = 0

                    #_, cost_val, lr = sess.run([optimizer, cost, learning_rate], feed_dict={X: batch_x, Y: batch_y_encode})
                    _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y_encode})
                    total_cost += cost_val
                    #print ("cost_val:", cost_val)

                #print('Epoch: %04d' % (epoch + 1), 'Avg. cost = {:.3f}'.format(total_cost / total_batch), 'learning_rate = {:.5f}'.format(lr))
                print('Epoch: %04d' % (epoch + 1), 'Avg. cost = {:.3f}'.format(total_cost / total_batch))#, 'learning_rate = {:.5f}'.format(lr))

                #TODO: training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_hand_shuffled = train_set_shuffled [:, train_data.shape[1]:]

                if (epoch + 1) % self.saved_period == 0 and epoch != 0:
                    model_path = self.model_dir + '/' + 'checkpoint_epoch'#_{}'.format(epoch) + '.ckpt'
                    save_path = saver.save(sess, model_path, global_step=global_step)
                    print('Model saved in file: %s' % save_path)

            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s):", stop_time - start_time)

            #model_path = self.model_dir + '/' + 'checkpoint_final.cpkt'
            #save_path = saver.save(sess, model_path)
            #print('Model saved in file: %s' % save_path)

    def test_nn(self):
        X, Y, logits, weights = self.build_model(self.dim_data, self.dim_hand, self.neuron_num, self.hidden_layer_num, self.epoch, self.batch_size)

        global_step = tf.Variable(0, trainable = False)

        """ Evaluate model """
        is_correct = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        init = tf.global_variables_initializer()

        """
        if self.model_name == None:
            print('Model does not exist')
            sys.exit(-1)
        """

        #print('test data shape: {}'.format(self.test_data.shape))

        #meta_graph_file = './checkpoint/checkpoint_epoch_4.ckpt-2475.meta'
        #model_file = './checkpoint/checkpoint_epoch-49500'

        model_file = self.model_dir + '/' + self.model_name

        #saver = tf.train.import_meta_graph(meta_graph_file)
        #ckpt_path = self.model_dir + '/' + self.model_name

        #print(ckpt_path)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            #test_output = sess.run(tf.argmax(logits, 1), feed_dict={X: self.test_data})
            #print (test_output)
            #sys.exit (-1)

            #TODO: create batch from test data
            #test_data -> all_possible_case

            # record the time when testing starts
            start_time = time.time()

            test_data = self.test_data
            test_hand = self.test_hand
            test_hand_encode = self.enc.transform (test_hand).toarray()
            test_set = np.concatenate ((test_data, test_hand), axis = 1)
            test_set_shuffled = np.random.permutation(test_set)
            test_data_shuffled = test_set_shuffled [:, 0:test_data.shape[1]]
            test_hand_shuffled = test_set_shuffled [:, test_data.shape[1]:]

            total_batch = int(len(self.train_data)/self.batch_size)

            total_correct_preds = 0
            index_counter = 0

            for i in range (total_batch):
                start_index = index_counter * self.batch_size
                end_index = (index_counter + 1) * self.batch_size
                batch_x = test_data_shuffled [start_index : end_index]
                batch_y = test_hand_shuffled [start_index : end_index]
                if batch_y.shape[0] == 0:
                    print ("Buzzz!")
                    print (start_index, end_index)
                    break
                batch_y_encode = self.enc.transform (batch_y).toarray() #batch_y#
                index_counter = index_counter + 1

                if (index_counter >= total_batch):
                    index_counter = 0

                #TODO: feed test data
                #TODO: use optimizer, cost???
                #_, cost_val, logits_batch = sess.run([optimizer, cost, logits], feed_dict={X: batch_x, Y: batch_y_encode})
                logits_batch = sess.run(logits, feed_dict={X: batch_x})

                preds = tf.nn.softmax(logits_batch)
                correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(batch_y_encode, 1))
                accuracy_float = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
                total_correct_preds += sess.run(accuracy_float)


            #TODO: find result
            accuracy_float = total_correct_preds / len (test_data)
            print ("Acuracy: ", accuracy_float)

            #print('--- Accuracy:', sess.run(accuracy, feed_dict={X: self.test_data, Y: test_hand_encode}))
            print ("test predicted hand:", sess.run (tf.argmax(tf.nn.softmax(logits), 1)[0:10], feed_dict={X: self.test_data}))
            print ("test ground truth hand:", sess.run (tf.argmax(Y, 1)[0:10], feed_dict={Y: test_hand_encode}))
            np.savetxt ("./output/output_task1.txt", sess.run (tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: self.test_data}), fmt = "%d")

            stop_time = time.time()
            print ("Testing time (s):", stop_time - start_time)

    def modify_hand(self):
        X, Y, logits, weights = self.build_model(self.dim_data, self.dim_hand, self.neuron_num, self.hidden_layer_num, self.epoch, self.batch_size)

        global_step = tf.Variable(0, trainable = False)

        """ Evaluate model """
        is_correct = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        init = tf.global_variables_initializer()

        """
        if self.model_name == None:
            print('Model does not exist')
            sys.exit(-1)
        """


        model_file = self.model_dir + '/' + self.model_name
        #meta_graph_file = './checkpoint/checkpoint_epoch_4.ckpt-2475.meta'
        #model_file = './checkpoint/checkpoint_epoch-49500'

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            #test_output = sess.run(tf.argmax(logits, 1), feed_dict={X: self.test_data})
            #print (test_output)
            #sys.exit (-1)

            #TODO: create batch from test data
            #test_data -> all_possible_case

            # record the time when testing starts
            start_time = time.time()

            test_data = self.test_data
            test_hand = self.test_hand
            test_hand_encode = self.enc.transform (test_hand).toarray()
            test_set = np.concatenate ((test_data, test_hand), axis = 1)
            test_set_shuffled = np.random.permutation(test_set)
            test_data_shuffled = test_set_shuffled [:, 0:test_data.shape[1]]
            test_hand_shuffled = test_set_shuffled [:, test_data.shape[1]:]
            modified_data = np.zeros ((0, 10))

            for i in range (len (test_data)):

                matrix = np.zeros ((0, 10))
                print ("Group card ", i)
                card1 = Card (test_data[i][0], test_data[i][1])
                card2 = Card (test_data[i][2], test_data[i][3])
                card3 = Card (test_data[i][4], test_data[i][5])
                card4 = Card (test_data[i][6], test_data[i][7])
                card5 = Card (test_data[i][8], test_data[i][9])

                list_card = [card1, card2, card3, card4, card5]

                a_group_card = GroupCard (list_card)
                predicted_hand = sess.run (tf.argmax(logits, 1), feed_dict={X: a_group_card.get_group_card_arr()})
                a_group_card.change_hand (predicted_hand)
                print (a_group_card)
                copy_group_card = a_group_card.copy ()
                if predicted_hand == 9:
                    #TODO: save this card
                    print ("No change")
                    modified_data = np.concatenate ((modified_data, a_group_card.get_group_card_arr()))
                    continue
                mem_list = []
                mem_lib = {}
                count = 0
                for j in range (5):
                    #print ("---------", j, predicted_hand[0])
                    for suit in range (1, 5):
                        for rank in range (1, 14):
                            count += 1
                            mem_lib[count] = (j, suit, rank)
                            new_card = Card (suit, rank)
                            if new_card.is_same_card (card1) or new_card.is_same_card (card2) or new_card.is_same_card (card3) or new_card.is_same_card (card4) or new_card.is_same_card (card5):
                                mem_list.append (count)
                            #else:
                            copy_group_card.modify_a_card (new_card, j)
                            matrix = np.concatenate ((matrix, copy_group_card.get_group_card_arr()))
                            copy_group_card = a_group_card.copy ()

                #print (matrix)
                #print (mem_list)
                #for i in range (len (mem_list)):
                #    print (mem_lib[mem_list[i]])
                #TODO: feed test data

                preds = sess.run (tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: matrix})
                #print (preds)
                #sys.exit (-1)
                max_hand = predicted_hand[0]
                selected_index = -1
                for j in range (len (preds)):
                    if (preds[j] > max_hand) and (j not in mem_list):
                    #if (preds[j] > max_hand):
                        #print ("exist")
                        #if (j not in mem_list):
                            #print ("found")
                        selected_index = j
                        max_hand = preds[j]
                print (selected_index, max_hand)
                if selected_index > -1:
                    position = mem_lib[selected_index][0]
                    chosen_card = Card (mem_lib[selected_index][1], mem_lib[selected_index][2])
                    #new_card = Card (0, 1)

                    #print (position)
                    #print (chosen_card)

                    #print (copy_group_card.get_group_card_arr())
                    copy_group_card.modify_a_card (chosen_card, position)
                    #print (copy_group_card.get_group_card_arr())
                    modified_data = np.concatenate ((modified_data, copy_group_card.get_group_card_arr()))
                else:
                    modified_data = np.concatenate ((modified_data, a_group_card.get_group_card_arr()))

                    #sys.exit(-1)

            #TODO: save new group cards into this file
            np.savetxt ("./output/output_task2.txt", modified_data, fmt = "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d")

            stop_time = time.time()
            print ("Changing hand time (s):", stop_time - start_time)
    def test_encode (self):

        batch_y =self.train_hand [0 : 2]
        print (batch_y)
        batch_y_encode = self.enc.transform (batch_y).toarray()
        print (batch_y_encode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #configuration
    parser.add_argument('--gpu_idx', type=str, default = '0')
    parser.add_argument('--model_dir', type=str, default = './checkpoint')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--saved_period', type=int, default=50, help='save checkpoint per X epoch')

    #hyper parameter
    parser.add_argument('--epoch', type=int, default = 2000)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default=0.00125)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--decay_step', type=int, default=2000)

    #network parameter
    parser.add_argument('--dim_data', type=int, default=10)
    parser.add_argument('--dim_hand', type=int, default=10)
    parser.add_argument('--hidden_layer_num', type=int, default = 2)
    parser.add_argument('--neuron_num', type=int, default = 1000)

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists('./output'):
        os.makedirs('./output')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx

    task1 = Task1 (train_file, test_data_file, test_label_file, args)

    if args.mode == 0:
        #task1.test()
        task1.train_nn()
    elif args.mode == 1:
        task1.test_nn()
    elif args.mode == 2:
        task1.modify_hand()
    else:
        print('wrong mode')
        sys.exit(-1)
"""
            test_hand_encode = self.enc.transform (self.test_hand).toarray() #self.test_hand#
            print('accuracy:', sess.run(accuracy, feed_dict={X: self.test_data, Y: test_hand_encode, dropout:self.DROP_OUT}))

            print ("test predicted hand:", sess.run (tf.argmax(prediction, 1)[0:10], feed_dict={X: self.test_data, dropout:self.DROP_OUT}))
            print ("test ground truth hand:", sess.run (tf.argmax(Y, 1)[0:10], feed_dict={Y: test_hand_encode}))
            np.savetxt ("output_task1.txt", sess.run (tf.argmax(prediction, 1), feed_dict={X: self.test_data, dropout:self.DROP_OUT}), fmt = "%d")
            """

"""
            test_hand_encode = self.enc.transform (self.test_hand).toarray() #self.test_hand#
            print('accuracy:', sess.run(accuracy, feed_dict={X: self.test_data, Y: test_hand_encode, dropout:self.DROP_OUT}))

            print ("test predicted hand:", sess.run (tf.argmax(prediction, 1)[0:10], feed_dict={X: self.test_data, dropout:self.DROP_OUT}))
            print ("test ground truth hand:", sess.run (tf.argmax(Y, 1)[0:10], feed_dict={Y: test_hand_encode}))
            np.savetxt ("output_task1.txt", sess.run (tf.argmax(prediction, 1), feed_dict={X: self.test_data, dropout:self.DROP_OUT}), fmt = "%d")
"""

