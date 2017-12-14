import tensorflow as tf
import csv
import numpy as np

RATE_TRAIN = "ee412.train"

TEST_DATA_SIZE = 10000

def load_dataset(is_train=True):
    
    with open(RATE_TRAIN) as f:
            data_read = csv.reader(f, delimiter="\t")
            data_list = list(data_read)
	    data_len = len(data_list)
	    train_data_end = data_len - TEST_DATA_SIZE
	    if (is_train):
		    data_list = data_list[:train_data_end]
	    else:
		    data_list = data_list[train_data_end:]
    #	print (data_list)

    data_features = np.array(data_list)[:,[0,1,3]]
    data_features = data_features.astype(np.float32)
    data_labels = np.array(data_list)[:,[2]]
    data_labels = data_labels.astype(np.float32)

    return data_features, data_labels

#features, labels = load_dataset()
#print (features.shape,labels.shape)

