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

from datamanager import PreProcess, GetDatasetInfo
from pret_models import Get_PreTrainedModel

#
# Global variables
#

K.set_image_dim_ordering('tf')

#
# Main
#

if __name__ == '__main__':

	#
	# Path for training and validation data
	#

	data_folder = '.'
	train_data_file = os.path.join(data_folder,'dataset-test-256.pkl')
	valid_data_file = os.path.join(data_folder,'dataset-test-256.pkl')

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
	# Log folder
	#

	if not os.path.exists('./log'):
		os.makedirs('./log')

	model_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

	boardfolder = os.path.join('./log',model_id)

	os.mkdir(boardfolder)

	print('Tensorboard folder: '+boardfolder)

	CallBacks = [TensorBoard(log_dir=boardfolder, write_graph=False)]

	#
	# Model folder
	#

	if not os.path.exists('./models'):
		os.makedirs('./models')

	model_folder = os.path.join('./models',model_id)

	os.mkdir(model_folder)

	print('Model folder: '+model_folder)

	#
	# Export useful information
	#

	n_batches_train, batch_size_train, Classes, im_size = GetDatasetInfo(train_data_file)
	n_batches_valid, batch_size_valid, Classes, im_size = GetDatasetInfo(valid_data_file)

	with open(os.path.join(model_folder,'model.info'), 'w') as fp:
		json.dump({'Classes': list(Classes),
				   'NBatchesTrain': n_batches_train,
				   'NBatchesValid': n_batches_valid,
			       'BatchSizeTrain': batch_size_train,
			       'BatchSizeValid': batch_size_valid,
			       'InputSize': im_size,
			       'LogDirectory': model_id,
			       'Model': model_name,
			       'Epochs': nepochs}, fp)

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

	for fold in range(crossval):

		best_acc = 0.

		checkpoint_name = 'model-'+str(fold)

		#
		# Create the model
		#

		model = Get_PreTrainedModel(model_name=model_name, im_size=im_size, n_classes=len(Classes))

		#
		# Save model to disk
		#

		model_json = model.to_json()
		with open(os.path.join(model_folder,'model.json'), 'w') as json_file:
			json_file.write(model_json)

		#
		# Loop the entire dataset nepochs times 
		#

		for epoch in range(nepochs):

			print('Epoch: '+str(epoch)+', Fold: '+str(fold))

			#
			# Training for each batch in the pickle file
			#

			with open(train_data_file, 'rb') as pickle_file_train:

				while 1:
				
					new_batch = False

					try:
				
						Batch = pickle.load(pickle_file_train)

						new_batch = True

					except EOFError:

						break

					if new_batch:

						XTrain, YTrain, Classes, (n_samples,n_classes) = PreProcess(Batch=Batch, model_name=model_name)

						DataGen.fit(XTrain)

						model.fit_generator(DataGen.flow(XTrain, YTrain, batch_size=32), steps_per_epoch=len(XTrain) / 32, epochs=1, verbose=0, callbacks=CallBacks)

			#
			# At the end of each epoch evaluate the model
			# on the validation set and decide whether to
			# save it
			#

			cur_acc = 0.

			with open(valid_data_file, 'rb') as pickle_file_valid:

				while 1:
				
					new_batch = False

					try:
				
						Batch = pickle.load(pickle_file_valid)

						new_batch = True

					except EOFError:

						break

					if new_batch:

						XValid, YValid, _, _ = PreProcess(Batch=Batch, model_name=model_name)

						metrics = model.evaluate(XValid, YValid, batch_size=32, verbose=0)

						cur_acc += metrics[1]
			
			cur_acc /= n_batches_valid

			if cur_acc > best_acc:

				print('Saving new model...')
				print('Old acc:'+str(best_acc))
				print('New acc:'+str(cur_acc))

				best_acc = cur_acc

				model.save_weights(os.path.join(model_folder,checkpoint_name+'.h5'))

		#
		# Clear session and start over a new fold
		#

		K.clear_session()
