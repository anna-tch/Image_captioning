#!/usr/bin/python3
# -*- coding:utf-8 -*-

import time
import os 
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import KFold, GroupKFold
from time import gmtime, strftime


# modules
from preprocess_captions import *
from preprocess_images import *
from model import *












def main():

	print('\n\n\n#######################################################################')
	print('#  Preprocess captions ')
	print('#######################################################################\n\n\n')


	# load file
	doc = load_doc("../captions.txt")

	# store in a dict
	descriptions = load_descriptions(doc)

	# convert to lists
	all_captions, all_img_name_vector= to_lines(descriptions)

	print("Len all captions : {}\nLen all ids : {}\n".format(len(all_captions), len(all_img_name_vector)))
	print("Example of a caption : {} \nExample of an image id : {}".format(all_captions[0], all_img_name_vector[0]) )

	# encode text
	tokenizer, all_seqs, cap_vector = tokenize_and_encode(all_captions)
	# Calculate the max_length, which is used to store the attention weights
	max_length = calc_max_length(all_seqs)
	top_k = len(tokenizer.word_index)
	vocab_size = top_k + 1


	print("Len vocab : {}".format(len(tokenizer.word_index)))
	print("Sequence : {}".format(all_seqs[0]))
	print("Padded sequence : {}".format(cap_vector[0]))
	print("Max length : {}".format(max_length))




	print('\n\n\n#######################################################################')
	print('#  Split to train and test datasets ')
	print('#######################################################################\n\n\n')


	# splits data so that all of the image dublicates corresponding to the five captions 
	# would stay at one set
	kf = GroupKFold(n_splits=10).split(X=all_img_name_vector, groups=all_img_name_vector)

	for ind, (tr, test) in enumerate(kf):
		img_name_train = np.array(all_img_name_vector)[tr] # np.array make indexing possible
		img_name_test = np.array(all_img_name_vector)[test]
	
		cap_train =  cap_vector[tr]
		cap_test =  cap_vector[test]
		break



	#print(img_name_train[:6],'\n')
	#print(cap_train[:6],'\n')
	print("All : ", len(all_img_name_vector), len(cap_vector))
	print("Train : ", len(img_name_train), len(cap_train))
	print("Test : ", len(img_name_test), len(cap_test))




	print('\n\n\n#######################################################################')
	print('#  Preprocess images ')
	print('#######################################################################\n\n\n')


	AUTO = tf.data.experimental.AUTOTUNE


	# define train image dataset
	encode_train = sorted(set(img_name_train))
	train_image_dataset = (tf.data.Dataset
				 .from_tensor_slices(encode_train)
				 .map(load_image, num_parallel_calls=AUTO)
				 .map(data_augment, num_parallel_calls=AUTO)
				 .batch(16)
				 .prefetch(AUTO))

	# define test image dataset
	encode_test = sorted(set(img_name_test))
	test_image_dataset = (tf.data.Dataset
				 .from_tensor_slices(encode_test)
				 .map(load_image, num_parallel_calls=AUTO)
				 .batch(16)
				 .prefetch(AUTO))

	# count numpy files with features
	max_images = 1291
	cnt_npy = 0
	for fname in os.listdir('../images'):
		if fname.endswith('.npy'):
			cnt_npy +=1
			
	
	# if not all image features have been extracted, then
	if cnt_npy != max_images:

		# get the extraction model
		inception_model = get_inception_model()

		# extract train features
		extract_features(train_image_dataset, inception_model)

		# extract test features
		extract_features(test_image_dataset, inception_model)



	print('\n\n\n#######################################################################')
	print('#  Define model ')
	print('#######################################################################\n\n\n')

	BATCH_SIZE = 64
	BUFFER_SIZE = 1000
	embedding_dim = 256
	units = 512
	num_steps = len(img_name_train) // BATCH_SIZE
	features_shape = 2048
	attention_features_shape = 64




	try :
		train_dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

		# Use map to load the numpy files in parallel
		train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(
										map_func, [item1, item2], [tf.float32, tf.int32]),
										num_parallel_calls=AUTO)

		# Shuffle and batch
		train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
		train_dataset = train_dataset.prefetch(buffer_size=AUTO)

		# create encoder and decoder
		encoder = CNN_Encoder(embedding_dim)
		decoder = RNN_Decoder(embedding_dim, units, vocab_size)

		# set optimizer and loss
		optimizer = tf.keras.optimizers.Adam()
		loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

		print("Done!")

	except:
		print("Not done!")




	print('\n\n\n#######################################################################')
	print('#  Train model ')
	print('#######################################################################\n\n\n')


	# set checkpoint
	start_epoch = 0
	checkpoint_path = "./checkpoints/train"
	ckpt = tf.train.Checkpoint(encoder=encoder,
						   decoder=decoder,
						   optimizer = optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
	if ckpt_manager.latest_checkpoint:
		start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
		# restoring the latest checkpoint in checkpoint_path
		ckpt.restore(ckpt_manager.latest_checkpoint)


	loss_plot = []

	EPOCHS = 120

	for epoch in range(start_epoch, EPOCHS):
		start = time.time()
		total_loss = 0

		# train
		for (batch, (img_tensor, target)) in enumerate(train_dataset):
			batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, tokenizer)
			total_loss += t_loss

			if batch % 100 == 0:
				print ('Epoch {} Batch {} Loss {:.4f}'.format(
				epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
		# storing the epoch end loss value to plot later
		loss_plot.append(total_loss / num_steps)


		if epoch % 5 == 0:
			ckpt_manager.save()

	print ('Epoch {} Loss {:.6f}'.format(epoch + 1,total_loss/num_steps))
	print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


	print('\n\n\n#######################################################################')
	print('#  Test model ')
	print('#######################################################################\n\n\n')

	# test the model on the new images
	all_results = []
	for image_id in tqdm(img_name_test):
		results = {}
		# id
		idx = image_id.split("/")[-1].split(".")[0]
		results["id"] = idx

		# original captions
		real_caption = [caption for caption in descriptions[idx]]
		results["real"] = real_caption

		# predicted caption
		predicted, attention_plot = evaluate(image_id, encoder, decoder, tokenizer, max_length)
		predicted = " ".join(predicted).replace("<end>", "")
		results["predicted"]= predicted

		# save
		all_results.append(results)


	# to lines
	actual = []
	predicted = []
	ids = []

	for dico in all_results:
		ids.append(dico['id'])
		actual.append(dico['real'])
		predicted.append(dico['predicted'])

	# save the results
	t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
	df = pd.DataFrame(actual, columns=['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5'])
	df.insert(0, "image_id", ids, True)
	df.insert(1, "predicted", predicted, True)
	df.to_csv('results{}.csv'.format(t), encoding='utf-8', sep = '\t')


	# calculate BLEU score
	b1=corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
	b2=corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
	b3=corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
	b4=corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

	print('BLEU-1: %f' % b1)
	print('BLEU-2: %f' % b2)
	print('BLEU-3: %f' % b3)
	print('BLEU-4: %f' % b4)



	return True








if __name__ == '__main__':
	main()