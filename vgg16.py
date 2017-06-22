import os
import sys
import pickle
import getopt
import warnings
import datetime
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils.data_utils import get_file

from Aux import LoadDataset, SplitData

#
# Global variables
#

K.set_image_dim_ordering('tf')

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


#
# Get command line arguments
#

opts, _ = getopt.getopt(sys.argv[1:],'c:e:',['crossval=','epochs='])

for opt, arg in opts:
	if opt in ('-e','--epochs'):
		nepochs = np.int(arg)
	if opt in ('-c','--crossval'):
		crossval = np.int(arg)

#
# Load data
#

X, Y, Classes, (n_samples,n_classes) = LoadDataset('Dataset.pkl')

print('#Classes: '+str(n_classes)+', #Samples: '+str(n_samples))

#
# Load Model
#

def VGG16(include_top=True, weights='imagenet', input_shape=None, pooling=None, classes=1000):

	img_input = Input(shape=(None,None,3))

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	model = Model(img_input, x, name='vgg16')

	weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models')
	model.load_weights(weights_path)

	return model

if __name__ == '__main__':

	#
	# Log folder
	#

	if not os.path.exists('./log'):
		os.makedirs('./log')

	logname = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

	boardfolder = os.path.join('./log',logname)

	os.mkdir(boardfolder)

	print('Tensorboard folder: '+boardfolder)


	#
	# Callbacks
	#

	CallBacks = [ModelCheckpoint('VGG16.h5', monitor='val_loss', save_best_only=True),
				 #EarlyStopping(monitor='val_loss',mode='auto'),
				 TensorBoard(log_dir=boardfolder, write_graph=False)]

	#
	# VGG16 Model adjustments for new dataset
	#

	base_model = VGG16(include_top=False, weights='imagenet')

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	for layer in base_model.layers:
		layer.trainable = False

	#
	# Training
	#

	print('Training in '+str(crossval)+' folds for '+str(nepochs)+' epochs')

	for fold in range(crossval):

		XTrain, YTrain, XTest, YTest = SplitData(X, Y, n_samples, n_classes, split_fac=0.10)

		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

		model.fit(XTrain, YTrain, epochs=nepochs, batch_size=32, shuffle=True, verbose=1, validation_data=(XTest,YTest), callbacks=CallBacks)

	K.clear_session()
