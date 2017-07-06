import pickle
import numpy as np

def LoadDataset(full_path):

	with open(full_path, 'rb') as fp:
		Dataset = pickle.load(fp)

	Data = Dataset['Data'].astype(np.float64)
	Labels = Dataset['Label']

	Classes = Labels[np.sort(np.unique(Labels, return_index=True)[1])]

	n_classes = len(Classes)
	n_samples = Data.shape[0]

	for group in range(n_classes):
		Labels[Labels==Classes[group]] = group
	Labels = Labels.astype(np.int)

	Codes = np.zeros((len(Labels), n_classes))
	Codes[np.arange(len(Labels)), Labels] = 1

	for channel in range(3):
		Data[:,:,:,channel] -= np.mean(Data[:,:,:,channel])

	return Data, Codes, Classes, (n_samples,n_classes)

def SplitData(Data, Codes, n_samples, n_classes, split_fac=0.25):

	Ids = np.random.choice(np.arange(n_samples), size=n_samples, replace=False)

	XTrain = Data[Ids[0:int((1-split_fac)*Data.shape[0])],:,:,:]

	XTest = Data[Ids[int((1-split_fac)*Data.shape[0]):Data.shape[0]],:,:,:]

	YTrain = Codes[Ids[0:int((1-split_fac)*Data.shape[0])]]

	YTest = Codes[Ids[int((1-split_fac)*Data.shape[0]):Data.shape[0]]]

	return XTrain, YTrain, XTest, YTest
