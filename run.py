import tensorflow as tf
import numpy as np
from model.DeepModel import DeepModel
from util.dataloader import construct_tf_dataset

# graph
graph = tf.Graph()
sess = tf.Session(graph=graph)

# constructing tf dataset
with graph.as_default():
	train_dataset, field_size, feature_size = construct_tf_dataset('data/train_data', 'data/train_feature_dict')
	train_iter = train_dataset.make_one_shot_iterator()
	next_batch_op = train_iter.get_next()

# model
args = {
	'field_size': field_size,
	'feature_size': feature_size,
	'embedding_size': 8,
	'layers': [32, 32],
	'learning_rate': 3e-4
}
model = DeepModel(args, graph, sess)
model.init_weights()

next_batch = sess.run(next_batch_op)
print(next_batch)

features = next_batch['features']
label = next_batch['label'] 
label = np.reshape(label, [-1, 1])

log_loss = model.eval(features, label)
print('log_loss: ', log_loss)