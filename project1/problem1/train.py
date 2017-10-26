import tensorflow as tf
import csv
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data

f = open('train_data.csv', 'r')
test_file = open('test_data.csv', 'r')
test_label_file = open('test_hand.csv', 'r')

rdr = csv.reader(f)
rdr_test = csv.reader(test_file)
rdr_test_label = csv.reader(test_label_file)

ori_train_set = []
test_set = []
tmp_test_label_set = []

for line in rdr:
    # print(line)
    ori_train_set.append(map(int, line))
f.close()

for line in rdr_test:
	test_set.append(map(int, line))
test_file.close()

for line in rdr_test_label:
	tmp_test_label_set.append(map(int, line))
test_label_file.close()

ori_train_set = np.array(ori_train_set)
train_set = np.random.permutation(ori_train_set)

test_set = np.array(test_set)
tmp_test_label_set = np.array(tmp_test_label_set)

input_set = train_set[:,0:10]
label_set = np.zeros((len(input_set),10))
tmp_label_set = train_set[:,10:11]

test_label_set = np.zeros((len(test_set), 10))

#to make train label set
for i in range(len(input_set)):
	label_set[i,tmp_label_set[i]] = 1
#to make test label set
for i in range(len(test_set)):
	test_label_set[i,tmp_test_label_set[i]] = 1

print ((input_set))
print (label_set)

# mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 10])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([10, 1000], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([1000, 1000], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([1000, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
# total_batch = int(mnist.train.num_examples / batch_size)
total_batch = int(len(train_set) / batch_size)
print(total_batch)

index_counter = 0
def next_batch(f_batch_size):
	global index_counter, total_batch
	start_index = index_counter * f_batch_size
	end_index = (index_counter +1) * f_batch_size
	my_batch_x = input_set[start_index : end_index]
	my_batch_y = label_set[start_index : end_index]
	index_counter = index_counter + 1
	if (index_counter >= total_batch):
		index_counter = 0
	return (my_batch_x, my_batch_y)

for epoch in range(100):
    total_cost = 0

    for i in range(total_batch):
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs, batch_ys = next_batch(batch_size)

        # print (np.shape(batch_xs), np.shape(batch_ys))
        # print (batch_xs.size, batch_ys.size)
        # exit(0)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

    #training data permutation
    train_set = np.random.permutation(ori_train_set)
    input_set = train_set[:,0:10]
    label_set = np.zeros((len(input_set),10))
    tmp_label_set = train_set[:,10:11]
    #to make train label set
    for i in range(len(input_set)):
    	label_set[i,tmp_label_set[i]] = 1

# print ("save model ...")

# saver = tf.train.Saver()
# save_path = saver.save(sess,"./model.ckpt")

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print (mnist.test.images)
# print('accuracy:', sess.run(accuracy,
#                         feed_dict={X: mnist.test.images,
#                                    Y: mnist.test.labels}))
print('accuracy:', sess.run(accuracy,
                        feed_dict={X: test_set,
                                   Y: test_label_set}))