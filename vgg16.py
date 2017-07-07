import os
import sys
import json
import pickle
import getopt
import warnings
import datetime
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, History
from keras.preprocessing.image import ImageDataGenerator
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

def VGG16(n_classes = 5):

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

	base_model = Model(img_input, x, name='vgg16')

	weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models')
	base_model.load_weights(weights_path)

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions, name='vgg16-adapted')

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
	# Training adapted VGG16 model
	#

	sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.5, nesterov=True)

	print('Training in '+str(crossval)+' folds for '+str(nepochs)+' epochs')

	Metrics = []

	for fold in range(crossval):

		#
		# Callbacks
		#

		checkpoint_name = 'vgg16-'+str(fold)+'.h5'

		CallBacks = [ModelCheckpoint(os.path.join(modelfolder,checkpoint_name), monitor='val_loss', save_best_only=True),
					 #EarlyStopping(monitor='val_loss',mode='auto'),
					 TensorBoard(log_dir=boardfolder, write_graph=False)]

		#
		# Training
		#

		model = VGG16(n_classes = n_classes)

		XTrain, YTrain, XTest, YTest = SplitData(X, Y, n_samples, n_classes, split_fac=0.10)

		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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
	with open(os.path.join(modelfolder,'vgg16.json'), 'w') as json_file:
		json_file.write(model_json)

	#
	# Export useful information
	#

	with open(os.path.join(modelfolder,'vgg16.info'), 'w') as fp:
		json.dump({'Classes': list(Classes), 'NSamples': n_samples, 'InputSize': X.shape[1], 'LogDirectory': modelname}, fp)
