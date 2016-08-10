import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D

def build_model(full_image_size):
	model = Sequential()
	model.add(Convolution2D(1,1,1, border_mode='same',  activation='sigmoid',input_shape = full_image_size))
	model.add(Convolution2D(10,5,5, border_mode='same', activation='relu'))
	model.add(Convolution2D(1,1,1, border_mode='same',  activation='sigmoid'))
	return model