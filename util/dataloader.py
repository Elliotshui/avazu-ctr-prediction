import numpy as np
import tensorflow as tf

import pickle

BATCH_SIZE = 256 
SHUFFLE_BUFFER_SIZE = 512
EPOCH = 5

def construct_tf_dataset(data_path, feature_dict_path):
	print("loading data...")
	train_data = pickle.load(open(data_path, 'rb'))
	feature_dict = pickle.load(open(feature_dict_path, 'rb'))
	print("========= data loaded ========")

	field_size = len(train_data['features'][0])
	feature_size = len(feature_dict)
	record_num = len(train_data['features'])
	print("field_size: ", field_size)
	print("feature_size: ", feature_size)
	print("record_number: ", record_num)

	train_size = int(0.8 * record_num)

	print("constructing tf dataset...")
	dataset = tf.data.Dataset.from_tensor_slices(train_data)
	dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
	train_set = dataset.take(train_size)
	validation_set = dataset.skip(train_size)

	train_set = train_set.batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).repeat(EPOCH)
	validation_set = validation_set.batch(BATCH_SIZE)
	print("======== tf dataset constructed ========")


	return train_set, validation_set, field_size, feature_size 

