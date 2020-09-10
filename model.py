
#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from preprocess_images import *



BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
features_shape = 2048
attention_features_shape = 64



optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
inception_model = get_inception_model()


class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, features, hidden):
		# features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

		# hidden shape == (batch_size, hidden_size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden_size)
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, 64, hidden_size)
		score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

		# attention_weights shape == (batch_size, 64, 1)
		# you get 1 at the last axis because you are applying score to self.V
		attention_weights = tf.nn.softmax(self.V(score), axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
	# Since you have already extracted the features and dumped it using pickle
	# This encoder passes those features through a Fully connected layer
	def __init__(self, embedding_dim):
		super(CNN_Encoder, self).__init__()
		# shape after fc == (batch_size, 64, embedding_dim)
		self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x



class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
								   return_sequences=True,
								   return_state=True,
								   recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)
		self.attention = BahdanauAttention(self.units)

	def call(self, x, features, hidden):
		# defining attention as a separate model
		context_vector, attention_weights = self.attention(features, hidden)

		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)

		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

		# passing the concatenated vector to the GRU
		output, state = self.gru(x)

		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)

		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))

		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)

		return x, state, attention_weights
	

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))




def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)



@tf.function
def train_step(img_tensor, target, encoder, decoder, tokenizer):
	loss = 0

	# initializing the hidden state for each batch
	# because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

	with tf.GradientTape() as tape:
		features = encoder(img_tensor)

		for i in range(1, target.shape[1]):
			# passing the features through the decoder
			predictions, hidden, _ = decoder(dec_input, features, hidden)

			loss += loss_function(target[:, i], predictions)

			# using teacher forcing
			dec_input = tf.expand_dims(target[:, i], 1)

	total_loss = (loss / int(target.shape[1]))

	trainable_variables = encoder.trainable_variables + decoder.trainable_variables

	gradients = tape.gradient(loss, trainable_variables)

	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss


def evaluate(image, encoder, decoder, tokenizer, max_length):
	attention_plot = np.zeros((max_length, attention_features_shape))

	hidden = decoder.reset_state(batch_size=1)

	temp_input = tf.expand_dims(load_image(image)[0], 0)
	img_tensor_val = inception_model(temp_input)
	img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

	features = encoder(img_tensor_val)

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
	result = []

	for i in range(max_length):
		predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

		attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

		predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
		result.append(tokenizer.index_word[predicted_id])

		if tokenizer.index_word[predicted_id] == '<end>':
			return result, attention_plot

		dec_input = tf.expand_dims([predicted_id], 0)

	attention_plot = attention_plot[:len(result), :]
	return result, attention_plot
