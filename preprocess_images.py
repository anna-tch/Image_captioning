#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm



def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.resize(img, (299, 299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img, image_path

def data_augment(image, image_path=None):

	image = tf.image.random_flip_left_right(image)

	return image, image_path


def get_inception_model():
	image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
	image_features_extract_model = tf.keras.Model(image_model.input, image_model.layers[-1].output)

	return image_features_extract_model


def extract_features(dataset, image_features_extract_model):
	'''
	INPUT: tf dataset containing image ids 
	OUTPUT : saves extracted features to numpy files
	'''
	for img, path in tqdm(dataset):
		batch_features = image_features_extract_model(img)
		batch_features = tf.reshape(batch_features,
							  (batch_features.shape[0], -1, batch_features.shape[3]))

		for bf, p in zip(batch_features, path):
			path_of_feature = p.numpy().decode("utf-8")
			np.save(path_of_feature, bf.numpy())

	return True


# Load the numpy files
def map_func(img_name, cap):
	img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	return img_tensor, cap