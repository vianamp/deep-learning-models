import os
import sys
import json
import pickle
import getopt
import warnings
import datetime
import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, History

from aux import LoadDataset, SplitData
from pret_models import Get_PreTrainedModel

#
# Global variables
#

K.set_image_dim_ordering('tf')

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#
# Get command line arguments
#

opts, _ = getopt.getopt(sys.argv[1:],'c:e:m:',['crossval=','epochs=','model='])

for opt, arg in opts:
	if opt in ('-e','--epochs'):
		nepochs = np.int(arg)
	if opt in ('-c','--crossval'):
		crossval = np.int(arg)
	if opt in ('-m','--model'):
		model_name = arg

#
# Load data
#

X, Y, Classes, (n_samples,n_classes) = LoadDataset('Dataset.pkl', model_name=model_name)

print('Model: '+model_name)

print('#Classes: '+str(n_classes)+', #Samples: '+str(n_samples))

#
# Main
#

if __name__ == '__main__':

	#
	# Log folder
	#

	if not os.path.exists('./log'):
		os.makedirs('./log')

	model_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

	boardfolder = os.path.join('./log',model_id)

	os.mkdir(boardfolder)

	print('Tensorboard folder: '+boardfolder)

	#
	# Model folder
	#

	if not os.path.exists('./models'):
		os.makedirs('./models')

	model_folder = os.path.join('./models',model_id)

	os.mkdir(model_folder)

	print('Model folder: '+model_folder)

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
	# Training adapted model
	#

	print('Training in '+str(crossval)+' folds for '+str(nepochs)+' epochs')

	Metrics = []

	for fold in range(crossval):

		#
		# Callbacks
		#

		checkpoint_name = 'model-'+str(fold)

		CallBacks = [ModelCheckpoint(os.path.join(model_folder,checkpoint_name+'.h5'), monitor='val_loss', save_best_only=True),
					 TensorBoard(log_dir=boardfolder, write_graph=False)]

		#
		# Create the model
		#

		model = Get_PreTrainedModel(model_name=model_name, im_size=224, n_classes=3)

		#
		# Split the data
		#

		XTrain, YTrain, XTest, YTest = SplitData(X, Y, n_samples, n_classes, split_fac=0.10)

		#
		# Save model to disk
		#

		model_json = model.to_json()
		with open(os.path.join(model_folder,'model.json'), 'w') as json_file:
			json_file.write(model_json)

		#
		# Export useful information
		#

		with open(os.path.join(model_folder,'model.info'), 'w') as fp:
			json.dump({'Classes': list(Classes),
				       'NSamples': n_samples,
				       'InputSize': X.shape[1],
				       'LogDirectory': model_id,
				       'Model': model_name,
				       'Epochs': nepochs}, fp)

		# -----------------------------------------------------------------------------------------
		# Training
		# -----------------------------------------------------------------------------------------

		DataGen.fit(XTrain)

		history = model.fit_generator(DataGen.flow(XTrain, YTrain, batch_size=32), steps_per_epoch=len(XTrain) / 32, epochs=nepochs, verbose=1, validation_data=(XTest,YTest), callbacks=CallBacks)

		Metrics = np.append(Metrics, history.history)

		#
		# Save history
		#

		with open(os.path.join(model_folder,'model.pkl'), 'wb') as fp:
			pickle.dump(Metrics, fp)


		K.clear_session()
