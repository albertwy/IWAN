import argparse
import numpy as np
seed = 123
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support
import tqdm
import time

from models_context import get_model
from dataset_context import SarcasmDataset, collate_fn

from torch.utils.data import DataLoader

import pickle



def main(args):
	train_dataset = []
	test_dataset = []


	for dataset_index in tqdm.trange(6):
		sarcasm_test = DataLoader(SarcasmDataset("test", dataset_index, "all_train", args.p), batch_size=args.bz, shuffle=True, collate_fn=collate_fn)
		sarcasm_train = DataLoader(SarcasmDataset("all_train", dataset_index, "all_train", args.p), batch_size=args.bz, shuffle=True, collate_fn=collate_fn)

		final_model = get_model(args.model)

		if torch.cuda.is_available():
			final_model = final_model.cuda()

		final_model.load_state_dict(torch.load('{}_{}.pkl'.format(args.name, dataset_index)))


		final_model.eval()

		features = []
		true_labels = []

		for speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, labels in tqdm.tqdm(sarcasm_train):
			if torch.cuda.is_available():
				speaker_vectors = speaker_vectors.cuda()
				bert_context_reps = bert_context_reps.cuda()
				audio_context_reps = audio_context_reps.cuda()
				video_context_reps = video_context_reps.cuda()

				bert_global_reps = bert_global_reps.cuda()
				audio_global_reps = audio_global_reps.cuda()
				video_global_reps = video_global_reps.cuda()
				video_tensor = video_tensor.cuda()
				audio_tensor = audio_tensor.cuda()
				text_tensor = text_tensor.cuda()
				labels = labels.cuda()
				masks = masks.cuda()
				v_masks = v_masks.cuda()
				a_masks = a_masks.cuda()


				output = final_model(speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, get_feature=True)

				output = output.detach().cpu()				
				features.append(output.numpy())
				true_labels.append(labels.detach().cpu().numpy())

		train_dataset.append([np.concatenate(features), np.concatenate(true_labels)])



		features = []
		true_labels = []

		for speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, labels in tqdm.tqdm(sarcasm_test):
			if torch.cuda.is_available():
				speaker_vectors = speaker_vectors.cuda()
				bert_context_reps = bert_context_reps.cuda()
				audio_context_reps = audio_context_reps.cuda()
				video_context_reps = video_context_reps.cuda()
				bert_global_reps = bert_global_reps.cuda()
				audio_global_reps = audio_global_reps.cuda()
				video_global_reps = video_global_reps.cuda()
				video_tensor = video_tensor.cuda()
				audio_tensor = audio_tensor.cuda()
				text_tensor = text_tensor.cuda()
				labels = labels.cuda()
				masks = masks.cuda()
				v_masks = v_masks.cuda()
				a_masks = a_masks.cuda()


				output = final_model(speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, get_feature=True)
				

				output = output.detach().cpu()
				features.append(output.numpy())
				true_labels.append(labels.detach().cpu().numpy())

		test_dataset.append([np.concatenate(features), np.concatenate(true_labels)])


	with open("{}_dataset.pickle".format(args.name), "wb") as f:
		pickle.dump([train_dataset, test_dataset], f)
	





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

	parser.add_argument('--p', default=0.5, type=float,
	                     help='sentiment words')


	args = parser.parse_args()

	print(args)

	main(args)

	print(args)
