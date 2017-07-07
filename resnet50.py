import os
import sys
import json
import pickle
import getopt
import warnings
import datetime
import numpy as np
from keras import layers
from keras import optimizers
from keras.optimizers import SGD
import keras.backend as K
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, History
from keras.preprocessing.image import ImageDataGenerator

from Aux import LoadDataset, SplitData

#
# Global variables
#

K.set_image_dim_ordering('tf')

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

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

def identity_block(input_tensor, kernel_size, filters, stage, block):

	filters1, filters2, filters3 = filters

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

	filters1, filters2, filters3 = filters

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x

def ResNet50(im_size=224, n_classes=5):

	img_input = Input(shape=(im_size,im_size,3))

	x = ZeroPadding2D((3, 3))(img_input)
	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=3, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	x = AveragePooling2D((7, 7), name='avg_pool')(x)

	base_model = Model(img_input, x, name='resnet50')

	weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')
	base_model.load_weights(weights_path)

	x = base_model.output
	# x = GlobalAveragePooling2D()(x)
	# x = Dense(1024, activation='relu')(x)
	x = Flatten()(x)
	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions, name='resnet50-adapted')

	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

	for layer in base_model.layers:
		layer.trainable = False

	return model

if __name__ == '__main__':
	
	#
	# Log folder
	#

	if not os.path.exists('./log'):
		os.makedirs('./log')

	modelname = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

	boardfolder = os.path.join('./log',modelname)

	os.mkdir(boardfolder)

	print('Tensorboard folder: '+boardfolder)

	#
	# Model folder
	#

	if not os.path.exists('./models'):
		os.makedirs('./models')

	modelfolder = os.path.join('./models',modelname)

	os.mkdir(modelfolder)

	print('Model folder: '+modelfolder)

	#
	# Data augmentation
	#

	DataGen = ImageDataGenerator(
				featurewise_center = True,
				featurewise_std_normalization = True,
				width_shift_range = 0.2,
				height_shift_range = 0.2,
				horizontal_flip = True)

	#
	# Training adapted resnet50 model
	#

	print('Training in '+str(crossval)+' folds for '+str(nepochs)+' epochs')

	Metrics = []

	for fold in range(crossval):

		#
		# Callbacks
		#

		checkpoint_name = 'resnet50-'+str(fold)+'.h5'

		CallBacks = [ModelCheckpoint(os.path.join(modelfolder,checkpoint_name), monitor='val_loss', save_best_only=True),
					 #EarlyStopping(monitor='val_loss',mode='auto'),
					 TensorBoard(log_dir=boardfolder, write_graph=False)]

		#
		# Training
		#

		model = ResNet50(im_size=X.shape[1], n_classes=3)

		XTrain, YTrain, XTest, YTest = SplitData(X, Y, n_samples, n_classes, split_fac=0.10)

		DataGen.fit(XTrain)

		history = model.fit_generator(DataGen.flow(XTrain, YTrain, batch_size=32), steps_per_epoch=len(XTrain) / 32, epochs=nepochs, verbose=1, validation_data=(XTest,YTest), callbacks=CallBacks)

		Metrics = np.append(Metrics, history.history)

	K.clear_session()

	#
	# Save history
	#

	with open(boardfolder+'.pkl', 'wb') as fp:
		pickle.dump(Metrics, fp)

	#
	# Save model to disk
	#

	model_json = model.to_json()
	with open(os.path.join(modelfolder,'resnet50.json'), 'w') as json_file:
		json_file.write(model_json)

	#
	# Export useful information
	#

	with open(os.path.join(modelfolder,'resnet50.info'), 'w') as fp:
		json.dump({'Classes': list(Classes), 'NSamples': n_samples, 'InputSize': X.shape[1], 'LogDirectory': modelname}, fp)
