import tensorflow as tf
import csv
import numpy as np

RATE_TRAIN = "ee412.train"

def load_dataset(is_train=True):
    #f = open(RATE_TRAIN, 'w')
    with open(RATE_TRAIN) as f:
            data_read = csv.reader(f, delimiter="\t")
            data_list = list(data_read)
            #data_list = [int(i) for i in data_read]
            #print data_list
    #	print (data_list)

    data_features = np.array(data_list)[:,[0,1,3]]
    data_features = data_features.astype(np.float32)
    data_labels = np.array(data_list)[:,[2]]
    data_labels = data_labels.astype(np.float32)

    return data_features, data_labels

#print(type(data_labels[0]))
#assert data_features.shape[0] == data_labels.shape[0]

"""
dataset = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
batched_dataset = dataset.batch(4)
iterator = batched_dataset.make_one_shot_iterator()
next_example, next_label = iterator.get_next()
print((next_example, next_label))
"""
