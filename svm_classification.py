import argparse
import numpy as np
seed = 123
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import pickle

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import tqdm


def svm_train(train_input, train_output, C_value):
    clf = svm.SVC(C=C_value, gamma='scale', kernel='rbf')

    return clf.fit(train_input, train_output)



def main(args):

	dc, idc = args.dc, args.idc
	with open("{}_dataset.pickle".format(args.name), "rb") as f:
		train_dataset, test_dataset = pickle.load(f)

	weighted_precision, weighted_recall = [], []
	weighted_fscores = []

	c_values = [dc for i in range(5)]
	c_values.append(idc)


	results = []
	for dataset_index in tqdm.trange(6):
		model = svm_train(train_dataset[dataset_index][0], train_dataset[dataset_index][1], c_values[dataset_index])

		pred = model.predict(test_dataset[dataset_index][0])
		label = test_dataset[dataset_index][1]

		result_string = classification_report(label, pred, digits=3)
		print(confusion_matrix(label, pred))
		print(result_string)

		result = classification_report(label, pred, output_dict=True, digits=3)

		weighted_fscores.append(result["weighted avg"]["f1-score"])
		weighted_precision.append(result["weighted avg"]["precision"])
		weighted_recall.append(result["weighted avg"]["recall"])


	print("#"*20)
	print("Dependent-Avg :")
	print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(np.mean(weighted_precision[:5]),
                                                                                                 np.mean(weighted_recall[:5]),
                                                                                                 np.mean(weighted_fscores[:5])))
	print("#"*20)
	print("Independent-Avg :")
	print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(weighted_precision[-1],
                                                                                                 weighted_recall[-1],
                                                                                                 weighted_fscores[-1]))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Sarcasm')

	parser.add_argument('--epochs', default=100, type=int, metavar='N',
	                    help='number of total epochs to run')

	parser.add_argument('--bz', default=32, type=int,
	                    metavar='N', help='mini-batch size (default: 32)')

	# text model
	parser.add_argument('--lr', default=0.0005, type=float,
	                    metavar='LR', help='initial learning rate')


	parser.add_argument('--name', default="", type=str, help='model instance  name')

	parser.add_argument('--model', default="", help='model class name')

	parser.add_argument('--dc', default=1, type=int, help='svm c')
	parser.add_argument('--idc', default=1000, type=int, help='svm c')


	args = parser.parse_args()

	print(args)

	main(args)

	print(args)
