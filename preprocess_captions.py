#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf


# read the doc

def load_doc(filename):
	''' reads text file
	INPUT : document (txt)
	OUTPUT : text (str)
	'''
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


def load_descriptions(doc):
	'''
	INPUT : descriptions (str)
	OUTPUT : descriptions (dict)
	'''
	descriptions = {}
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) > 1:
			# get image id and description
			tokens = line.split()
			image_id, image_desc = tokens[0], tokens[1:]
			# save image id without extension
			image_id = image_id.split('.')[0]
			# convert description tokens back to string
			image_desc = ' '.join(image_desc)
			if image_id not in descriptions:
				descriptions[image_id] = list()
			descriptions[image_id].append(image_desc)
	return descriptions



def to_lines(descriptions):
	'''
	INPUT : descriptions(dict)
	OUTPUT : all descriptions (list), all image ids (list)
	'''

	# create empty lists
	all_ids = list()
	all_desc = list()

	# iterate over image ids
	for key in descriptions.keys():

		# iterate over all captions belonging to that image
		for caption in descriptions[key]:
			# add start and end tokens
			caption = caption = '<start> ' + caption + ' <end>'
			all_desc.append(caption)

			# get a full name
			full_name = '../images/' +key + '.jpg'
			all_ids.append(full_name)

	return all_desc, all_ids




def tokenize_and_encode(captions):
	'''
	INPUT : captions (list)
	OUTPUT : tokenizer, encoded sequences, padded sequences
	'''

	# tokenize
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
												  oov_token="<unk>",
												  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
	# get vocabulary
	tokenizer.fit_on_texts(captions)

	# index 0 is reserved for padding
	tokenizer.word_index['<pad>'] = 0
	tokenizer.index_word[0] = '<pad>'

	# Create the tokenized vectors
	train_seqs = tokenizer.texts_to_sequences(captions)

	# Pad each vector to the max_length of the captions, pad_sequences calculates it automatically
	cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

	return tokenizer, train_seqs, cap_vector





def calc_max_length(seqs):
	'''
	INPUT : encoded sequences (list)
	OUTPUT : max len (int)
	'''
	return max(len(s) for s in seqs)
