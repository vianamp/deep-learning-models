import pickle
import numpy as np
import pandas as pd

fname = '/Users/mviana/Desktop/deep-learning-models/log/20170626T103649'

with open(fname+'.pkl','rb') as pf:
	Data = pickle.load(pf)

FullTable = pd.DataFrame()
for crossval in range(len(Data)):
	
	subData = Data[crossval]

	nepochs = len(subData['acc'])

	trn_acc = subData['acc']
	val_acc = subData['val_acc']
	trn_los = subData['loss']
	val_los = subData['val_loss']
	indexes = np.repeat(crossval,nepochs)[0]
	epochid = np.arange(nepochs)

	Table = pd.DataFrame({'crossval':  indexes,
						  'epoch':     epochid,
		                  'train_acc': trn_acc,
		                  'val_acc':   val_acc,
		                  'train_loss':trn_los,
		                  'val_loss':  val_los})

	FullTable = FullTable.append(Table)

print(FullTable)

FullTable.to_csv(fname+'.csv', index=False)