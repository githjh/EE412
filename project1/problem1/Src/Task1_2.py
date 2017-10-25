# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:22:47 2017

@author: user
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
#import matplotlib.pyplot as plt
import time
#from multimethod import multimethod

# Read training data from csv train file
train_headers = ["suit1","rank1","suit2","rank2","suit3","rank3","suit4","rank4","suit5","rank5","hand"]
data_test_headers = ["suit1","rank1","suit2","rank2","suit3","rank3","suit4","rank4","suit5","rank5"]
hand_test_headers = ["hand"]

train_dtype_dict = {"suit1":int,"rank1":int,"suit2":int,"rank2":int,"suit3":int,"rank3":int,"suit4":int,"rank4":int,"suit5":int,"rank5":int,"hand":int}
data_test_dtype_dict = {"suit1":int,"rank1":int,"suit2":int,"rank2":int,"suit3":int,"rank3":int,"suit4":int,"rank4":int,"suit5":int,"rank5":int}
hand_test_dtype_dict = {"hand":int}

train_file = "../Data/train_data.csv"
test_data_file = "../Data/test_data.csv"
test_label_file = "../Data/test_hand.csv"

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
        return GroupCard (self.list_cards, self.hand)
    
    def modify_a_card (self, new_card, index):
        self.list_cards[index] = new_card
    
    def change_hand (self, hand):
        self.hand = hand
        
    def get_group_card (self):
        list_suit_rank = []
        for i in range (self.size_group):
            list_suit_rank += self.list_cards[i].get_card()
        return list_suit_rank
    
    def get_group_card_arr (self):
        return np.array (self.get_group_card()).reshape (1, 2 * self.size_group)
    
    def __str__ (self):
        return str (self.get_group_card_arr()) + "-" + str (self.hand)


  
class Task1(GroupCard):
    
    def __init__(self, train_file, test_data_file, test_label_file):
        self.train_file = train_file
        self.test_data_file = test_data_file
        self.test_label_file = test_label_file
        
        self.train_data = self.get_data (self.train_file, train_headers, train_dtype_dict)[0]
        self.train_hand = self.get_data (self.train_file, train_headers, train_dtype_dict)[1]
        self.test_data = self.get_data (self.test_data_file, data_test_headers, data_test_dtype_dict)
        self.test_hand = self.get_data (self.test_label_file, hand_test_headers, hand_test_dtype_dict)   
        self.enc = OneHotEncoder()
        self.enc.fit(np.array ([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]))
        
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
        
    
    def training (self, sess, train_data, train_hand, optimizer, cost, no_epoch, total_batch, batch_size, X, Y, dropout):
        
        # record the time when training starts
        start_time = time.time()
        
        train_data = self.train_data
        train_hand = self.train_hand
        train_set = np.concatenate ((train_data, train_hand), axis = 1)
        train_set_shuffled = np.random.permutation(train_set)
        train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
        train_hand_shuffled = train_set_shuffled [:, train_data.shape[1]:]
        
        for epoch in range (no_epoch):
            
            total_cost = 0
            index_counter = 0
            
            for i in range (total_batch):
                start_index = index_counter * batch_size
                end_index = (index_counter + 1) * batch_size
                batch_x = train_data_shuffled [start_index : end_index]
                batch_y = train_hand_shuffled [start_index : end_index]
                batch_y_encode = self.enc.transform (batch_y).toarray() #batch_y#
                #print ("batch_x", batch_x)
                #print ("batch_y", batch_y)
                index_counter = index_counter + 1

                if (index_counter >= total_batch):
                    index_counter = 0
        
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y_encode, dropout:self.DROP_OUT})
                total_cost += cost_val
                #print ("cost_val:", cost_val)
        
            print('Epoch: %04d' % (epoch + 1), 'Avg. cost = {:.3f}'.format(total_cost / total_batch))
        
            #TODO: training data permutation 
            train_set_shuffled = np.random.permutation(train_set)
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_hand_shuffled = train_set_shuffled [:, train_data.shape[1]:]
            
        print ("Training finished!")
        stop_time = time.time()
        print ("Training time (s):", stop_time - start_time)
    
    def build_model (self, dim_hand, no_unit_in_a_hidden_layer, train_data, train_hand, no_epoch, total_batch, batch_size):
        
        """ Create_model """
        X = tf.placeholder(tf.float32, [None, train_data.shape[1]])
        Y = tf.placeholder(tf.float32, [None, dim_hand])
        
        dropout = tf.placeholder(tf.float32, name='dropout')
        
        
        W1 = tf.Variable(tf.random_normal([train_data.shape[1], no_unit_in_a_hidden_layer], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(X, W1))
        
        W2 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, no_unit_in_a_hidden_layer], stddev=0.01))
        L2 = tf.nn.relu(tf.matmul(L1, W2))
        L2_hat = tf.nn.dropout(L2, dropout, name='relu_dropout')
        
        W3 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, dim_hand], stddev=0.01))
        logits = tf.matmul(L2_hat, W3)
        prediction = tf.nn.softmax(logits)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
        
        """ Evaluate model """
        is_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        
        """ Initialize the variables with default values"""
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            
            # Run the initializer
            sess.run(init)
                
            """ Start training, using training data"""
            self.training (sess, train_data, train_hand, optimizer, cost, no_epoch, total_batch, batch_size, X, Y, dropout)
            
            """ Testing the model on test data"""
            test_hand_encode = self.enc.transform (self.test_hand).toarray() #self.test_hand#
            print('accuracy:', sess.run(accuracy, feed_dict={X: self.test_data, Y: test_hand_encode, dropout:self.DROP_OUT}))
            
            #print ("test predicted hand:", sess.run (tf.argmax(prediction, 1)[0:10], feed_dict={X: self.test_data, dropout:self.DROP_OUT}))
            #print ("test ground truth hand:", sess.run (tf.argmax(Y, 1)[0:10], feed_dict={Y: test_hand_encode}))
            np.savetxt ("../Results/output_task1.txt", sess.run (tf.argmax(prediction, 1), feed_dict={X: self.test_data, dropout:self.DROP_OUT}), fmt = "%d")
            
            
            """ Modify a card"""
            test_data = self.test_data
            test_hand = self.test_hand
            for i in range (len (test_data)):
                card1 = Card (test_data[i][0], test_data[i][1])   
                card2 = Card (test_data[i][2], test_data[i][3])   
                card3 = Card (test_data[i][4], test_data[i][5])   
                card4 = Card (test_data[i][6], test_data[i][7])   
                card5 = Card (test_data[i][8], test_data[i][9])   
                    
                list_card = [card1, card2, card3, card4, card5]
                hand = test_hand[i][0]
                largest_hand = hand
                   
                a_group_card = GroupCard (list_card, hand)
                copy_group_card = a_group_card.copy ()
                           
                for j in range (5):
                    for suit in range (1, 5):
                        for rank in range (2, 15):
                            new_card = Card (suit, rank)
                            if new_card.is_same_card (card1) or new_card.is_same_card (card2) or new_card.is_same_card (card3) or new_card.is_same_card (card4) or new_card.is_same_card (card5):
                                break
                            else:
                                copy_group_card.modify_a_card (new_card, j)
                                predicted_hand = sess.run (tf.argmax(prediction, 1), feed_dict={X: copy_group_card, dropout:self.DROP_OUT})
                                if predicted_hand > largest_hand:
                                    largest_hand = predicted_hand
                                    modified_group_card = copy_group_card
                                    modified_group_card.change_hand (largest_hand)
                                copy_group_card = a_group_card.copy ()
                                    
                print (modified_group_card)
                
    def test_test_data (self):
        """ Modify a card"""
        test_data = self.test_data
        #test_hand = self.test_hand
        print (test_data.shape)
        for i in range (test_data.shape[0]):
            card1 = Card (test_data[i][0], test_data[i][1])   
            card2 = Card (test_data[i][2], test_data[i][3])   
            card3 = Card (test_data[i][4], test_data[i][5])   
            card4 = Card (test_data[i][6], test_data[i][7])   
            card5 = Card (test_data[i][8], test_data[i][9])   
                
            list_card = [card1, card2, card3, card4, card5]
            
            a_group_card = GroupCard (list_card)
            #print (a_group_card)
            if i == 3421:
                print ("[%d] " %(i), a_group_card)
    
    def test_a_grp_card(self, sess, prediction, X, dropout):
        """ Modify a card"""
        card1 = Card (0, 2)   
        card2 = Card (0, 3)
        card3 = Card (0, 4)
        card4 = Card (0, 5)
        card5 = Card (0, 6)
        
        list_card1 = [card1, card2, card3, card4, card5]
        
        groupCard1 = GroupCard (list_card1)
        test_grp_card = groupCard1.get_group_card_arr()
        
        print ("group card:", test_grp_card)
        print ("test predicted hand for this group cards:", sess.run (tf.argmax(prediction, 1), feed_dict={X: test_grp_card, dropout:self.DROP_OUT}))
    
    def test_modify_a_card (self):
        card1 = Card (0, 2)   
        card2 = Card (0, 3)
        card3 = Card (0, 4)
        card4 = Card (0, 5)
        card5 = Card (0, 6)
        
        list_card1 = [card1, card2, card3, card4, card5]
        
        groupCard1 = GroupCard (list_card1)
        #test_grp_card = groupCard1.get_group_card_arr()
        print (groupCard1)
        
        new_card = Card (1, 7)
        groupCard1.modify_a_card (new_card, 3)
        print (groupCard1)
    
    def test (self):
        X_train = self.train_data
        y_train = self.train_hand
        
        X_test = self.test_data
        y_test = self.test_hand
        
        print ("X_train:", X_train.shape)
        print ("y_train:", y_train.shape)
        
        print ("X_test[0]:", X_test[0])
        print ("y_test[0]:", y_test[0])

    def test_encode (self):
        
        batch_y =self.train_hand [0 : 2]
        print (batch_y)
        batch_y_encode = self.enc.transform (batch_y).toarray()
        print (batch_y_encode)
     
    def test_model (self):
        dim_hand = 10
        no_unit_in_a_hidden_layer = 1000
        train_data = self.train_data
        train_hand = self.train_hand
        no_epoch = 100#300
        batch_size = 100
        total_batch = int(len(self.train_data) / batch_size)
        
        
        self.build_model (dim_hand, no_unit_in_a_hidden_layer, train_data, train_hand, no_epoch, total_batch, batch_size)
    
task1 = Task1 (train_file, test_data_file, test_label_file)
task1.test_model ()

    
