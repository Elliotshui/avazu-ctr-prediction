import tensorflow as tf
import numpy as np

class DeepModel:

	def __init__(args, graph, session):
		
		# copy args
		self.graph = graph
		self.session = session
		self.args = args
		feature_size = args["feature_size"]
		field_size = args["field_size"]
		embedding_size = args["embedding_size"]
		layers = args["layers"]
		num_layers = len(layers)

		with self.graph.as_default():
			# model structure
			self.input = tf.placeholder(tf.int32, shape = [None, None])
			self.label = tf.placeholder(tf.float32, shape = [None, 1])
			self.embedding_w = tf.Variable(
				tf.random_normal([feature_size, embedding_size], 0.0, 0.1)
			)
			self.embedding = tf.nn.embedding_lookup(self.embedding_w, self.input)
			self.dense_out = dict()
			self.dense_out[0] = tf.layers.dense(
				inputs=self.embedding, 
				units=layers[0], 
				activation=tf.nn.leaky_relu
			)	
			for i in range(1, num_layers):
				self.dense_out[i] = tf.layers.dense(
					inputs=self.dense_out[i - 1],
					units=layers[i], 
					activation=tf.nn.leaky_relu
				)
			self.logits = tf.layers.dense(
				inputs=self.dense_out[num_layers - 1], 
				units=1,
				activation=None
			)
			self.pred = tf.nn.sigmoid(self.logits)

			# loss
			self.log_loss = tf.losses.log_loss(label, self.pred)
			self.loss = self.log_loss

			# optimizer
			self.optimizer = tf.train.AdamOptimizer(
				learning_rate=args["learning_rate"]
			)
			self.update = self.optimizer.minimize(loss)

	def train_batch(input, label):
		loss, _ = self.session.run(
			[self.loss, self.update], 
			feed_dict={self.input: input, self.label: label}
		)
		return loss
	
	def eval(input, label):
		log_loss = self.session.run(
			[self.log_loss],
			feed_dict={self.input: input, self.label: label}
		)
		return log_loss