import numpy as np
import tensorflow as tf

import pickle

BATCH_SIZE = 256 
SHUFFLE_BUFFER_SIZE = 512

def construct_tf_dataset(data_path, feature_dict_path):
	print("loading data...")
	train_data = pickle.load(open(data_path, 'rb'))
	feature_dict = pickle.load(open(feature_dict_path, 'rb'))
	print("========= data loaded ========")

	field_size = len(train_data['features'][0])
	feature_size = len(feature_dict)
	print("field_size: ", field_size)
	print("feature_size: ", feature_size)

	print("constructing tf dataset...")
	dataset = tf.data.Dataset.from_tensor_slices(train_data)
	dataset = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE)
	print("======== tf dataset constructed ========")


	return dataset, field_size, feature_size

