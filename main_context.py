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



def main(args):


	results = []
	for dataset_index in tqdm.trange(6):
		sarcasm_train = DataLoader(SarcasmDataset("train", dataset_index, "train", args.p), batch_size=args.bz, shuffle=True, collate_fn=collate_fn)
		sarcasm_dev = DataLoader(SarcasmDataset("dev", dataset_index, "train", args.p), batch_size=args.bz, shuffle=True, collate_fn=collate_fn)
		sarcasm_test = DataLoader(SarcasmDataset("test", dataset_index, "all_train", args.p), batch_size=args.bz, shuffle=True, collate_fn=collate_fn)

		# best_score = 0
		results_epoch = []
		epochs = args.epochs

		model = get_model(args.model)

		if torch.cuda.is_available():
			model = model.cuda()

		loss_fn = nn.CrossEntropyLoss()
		token_loss_fn = nn.CrossEntropyLoss(ignore_index=5)


		optimizer = optim.Adam(model.parameters(), lr=args.lr)

		for epoch in tqdm.trange(epochs):
			model.train()

			for speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, labels in tqdm.tqdm(sarcasm_train):
				if torch.cuda.is_available():
					speaker_vectors = speaker_vectors.cuda()
					bert_context_reps = bert_context_reps.cuda()
					audio_context_reps = audio_context_reps.cuda()
					video_context_reps = video_context_reps.cuda()
					audio_global_reps = audio_global_reps.cuda()
					video_global_reps = video_global_reps.cuda()
					bert_global_reps = bert_global_reps.cuda()
					video_tensor = video_tensor.cuda()
					audio_tensor = audio_tensor.cuda()
					text_tensor = text_tensor.cuda()
					masks = masks.cuda()
					v_masks = v_masks.cuda()
					a_masks = a_masks.cuda()
					labels = labels.cuda()



				output = model(speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks)
				loss = loss_fn(output, labels)


				loss.backward()	
				optimizer.step()
				optimizer.zero_grad()


			avg_loss, avg_accuracy, avg_fscore, class_report = eval(model, loss_fn, sarcasm_dev, args)
			print(" [DEV] \n")
			print(" loss : {}, accuracy : {}, f1-score :{} \n".format(avg_loss, avg_accuracy, avg_fscore))
			print(class_report)

			results_epoch.append(avg_fscore)


		
		final_epochs = np.argmax(np.array(results_epoch), 0)
		final_epochs += 1

		print("Epochs Max nums is {}\n".format(final_epochs))
		print("Start using All data \n")


		sarcasm_train = DataLoader(SarcasmDataset("all_train", dataset_index, "all_train", args.p), batch_size=args.bz, shuffle=True, collate_fn=collate_fn)

		final_model = get_model(args.model)

		if torch.cuda.is_available():
			final_model = final_model.cuda()

		loss_fn = nn.CrossEntropyLoss()
		token_loss_fn = nn.CrossEntropyLoss(ignore_index=5)

		optimizer = optim.Adam(final_model.parameters(), lr=args.lr)

		for epoch in tqdm.trange(final_epochs):
			final_model.train()

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



				output = final_model(speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks)
				loss = loss_fn(output, labels)

				loss.backward()	
				optimizer.step()
				optimizer.zero_grad()

		torch.save(final_model.state_dict(), '{}_{}.pkl'.format(args.name, dataset_index))

		avg_loss, avg_accuracy, avg_fscore, class_report = eval(final_model, loss_fn, sarcasm_test, args)

		print(" [TEST-{}] \n".format(dataset_index))
		print(" loss : {}, accuracy : {}, f1-score :{} \n".format(avg_loss, avg_accuracy, avg_fscore))
		print(class_report)


		results.append([avg_loss, avg_accuracy, avg_fscore])


	result = np.mean(np.array(results[:5]), 0)

	print("\n")
	print( "[Dependent-Final Result] \n")
	print(results[:5])
	print(" loss : {}, accuracy : {}, f1-score :{} \n".format(result[0], result[1], result[2]))
	print("\n")

	print("\n")
	print( "[Independent-Final Result] \n")
	print(results[-1])
	print(" loss : {}, accuracy : {}, f1-score :{} \n".format(results[-1][0], results[-1][1], results[-1][2]))
	print("\n")



def eval(model, loss_fn, sarcasm_dataset, args):
	model.eval()
	gds = []
	preds = []
	losses = []
	for speaker_vectors, bert_context_reps,  audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, labels in tqdm.tqdm(sarcasm_dataset):
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


		output = model(speaker_vectors, bert_context_reps, audio_context_reps, video_context_reps, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks)

		loss = loss_fn(output, labels)
		losses.append(loss.detach().cpu().numpy())		

		gds.append(labels.cpu().numpy())
		pred = torch.argmax(output,1)
		preds.append(pred.cpu().numpy())



	preds  = np.concatenate(preds)
	gds = np.concatenate(gds)

	avg_loss = round(np.average(losses),4)
	avg_accuracy = round(accuracy_score(gds,preds)*100,2)
	avg_fscore = round(f1_score(gds,preds,average='weighted')*100,2)
	class_report = classification_report(gds,preds,digits=4)

	return avg_loss, avg_accuracy, avg_fscore, class_report





if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Sarcasm')

	parser.add_argument('--epochs', default=100, type=int, metavar='N',
	                    help='number of total epochs to run')

	parser.add_argument('--bz', default=32, type=int,
	                    metavar='N', help='mini-batch size (default: 32)')

	# text model
	parser.add_argument('--lr', default=0.0001, type=float,
	                    metavar='LR', help='initial learning rate')

	parser.add_argument('--p', default=0.5, type=float,
	                     help='sentiment words')

	parser.add_argument('--name', default="", type=str, help='model instance  name')

	parser.add_argument('--model', default="", help='model class name')



	args = parser.parse_args()

	print(args)

	main(args)

	print(args)
