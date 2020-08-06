import tensorflow as tf
import numpy as np
import os
from model.DeepModel import DeepModel
from util.dataloader import construct_tf_dataset
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# graph
graph = tf.Graph()
sess = tf.Session(graph=graph)

# constructing tf dataset
with graph.as_default():
	train_set, valid_set, field_size, feature_size = construct_tf_dataset('data/train_data', 'data/train_feature_dict')
	train_iter = train_set.make_one_shot_iterator()
	valid_iter = valid_set.make_one_shot_iterator()
	next_train_batch_op = train_iter.get_next()
	next_valid_batch_op = valid_iter.get_next()

# model
args = {
	'field_size': field_size,
	'feature_size': feature_size,
	'embedding_size': 64,
	'layers': [32, 32],
	'learning_rate': 3e-4
}
model = DeepModel(args, graph, sess)
model.init_weights()

'''
loss = 0.0
for i in range(1, 10000):
	next_batch = sess.run(next_batch_op)
	features = next_batch['features']
	label = next_batch['label'] 
	label = np.reshape(label, [-1, 1])
	loss += model.train_batch(features, label)
	if i % 1000 == 0:
		print("batch ", i, " average loss: ", loss / 100)
		loss = 0.0
'''

num_epoch = 5
num_batch = int(600000 / 256)
loss = 0.0

# training
for i in tqdm(range(1, num_epoch * num_batch + 1)):
	next_batch = sess.run(next_train_batch_op)
	features = next_batch['features']
	label = next_batch['label']
	loss += model.train_batch(features, label)
	if i % num_batch == 0:
		print("epoch ", i / num_batch, " average loss: ", loss / num_batch)

# eval
loss = 0.0
eval_iter = 10000
for i in tqdm(range(1, eval_iter + 1)):
	next_batch = sess.run(next_eval_batch_op)
	features = next_batch['features']
	label = next_batch['label'] 
	loss += model.eval(features, label)
print("loss: ", loss / eval_iter)
