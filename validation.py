import os
import sys
import json
import pickle
import getopt
import datetime
import numpy as np
import keras.models
from keras.models import model_from_json
from Aux import LoadDataset
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

#
# Load model
#

json_file = open('vgg16.json','r')
model = model_from_json(json_file.read())
json_file.close()

model.load_weights('vgg16.h5')

with open('vgg16.info', 'r') as fp:
	Info = json.load(fp)

#
# Load dataset
#

X, Y, Classes, (n_samples,n_classes) = LoadDataset('Dataset.pkl')

print('#Classes: '+str(n_classes)+', #Samples: '+str(n_samples))

#
# Prediction
#

Z = model.predict(X, batch_size=32)

Z = np.argmax(Z,axis=1)

Y = np.argmax(Y,axis=1)

#
# Metrics
#

M = confusion_matrix(Y, Z)

print(Classes)

print('Confusion matrix:')
print(M)

acc = accuracy_score(Y, Z)

print('Accuracy: '+str(acc))

recall = recall_score(Y, Z, average='macro')  

print('Recall: '+str(recall))

