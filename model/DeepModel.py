import tensorflow as tf
import numpy as np

class DeepModel:

	def __init__(self, args, graph, session):
		
		# copy args
		self.graph = graph
		self.session = session
		self.args = args
		feature_size = args['feature_size']
		field_size = args['field_size']
		embedding_size = args['embedding_size']
		layers = args['layers']
		lr = args['learning_rate']
		num_layers = len(layers)

		with self.graph.as_default():
			# model structure
			self.input = tf.placeholder(tf.int32, shape = [None, None])
			self.label = tf.placeholder(tf.float32, shape = [None, 1])
			self.embedding_w = tf.Variable(
				tf.random_normal([feature_size, embedding_size], 0.0, 0.1)
			)
			self.embedding = tf.nn.embedding_lookup(self.embedding_w, self.input)
			self.embedding = tf.reshape(self.embedding, [-1, field_size * embedding_size])
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
			self.logits = tf.reshape(self.logits, [-1, 1])
			self.pred = tf.nn.sigmoid(self.logits)

			# loss
			self.log_loss = tf.losses.log_loss(self.label, self.pred)
			self.loss = self.log_loss

			# optimizer
			self.optimizer = tf.train.AdamOptimizer(
				learning_rate=lr
			)
			self.update = self.optimizer.minimize(self.loss)

			self.init = tf.global_variables_initializer()

	def init_weights(self):
		print("initializing weights...")
		self.session.run(self.init)
		
	def train_batch(self, input, label):
		loss, _ = self.session.run(
			[self.loss, self.update], 
			feed_dict={self.input: input, self.label: label}
		)
		return loss
	
	def eval(self, input, label):
		log_loss = self.session.run(
			self.log_loss,
			feed_dict={self.input: input, self.label: label}
		)
		return log_loss
