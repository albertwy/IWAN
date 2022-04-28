import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import json
import random


seed = 42
random.seed(seed)



def cal_dataset_stats(dataset, keys):
	
	v = []
	a = []
	for key in keys:
		if key not in dataset.keys():
			continue
		if dataset[key]["v"] is not None:
			for visual in dataset[key]["v"]:
				if visual.shape[0] > 0:
					v.append(visual)
		
		for acoustic in dataset[key]["a"]:
			if acoustic.shape[0] > 0:
				a.append(acoustic)

	mean_v = np.mean(np.concatenate(v, 0), 0)
	std_v = np.std(np.concatenate(v, 0), 0)


	mean_a = np.mean(np.concatenate(a, 0), 0)
	std_a = np.std(np.concatenate(a, 0), 0)


	return (0, 1, mean_a, std_a)
	# return (mean_v, std_v, mean_a, std_a)




with open("SentiWords_1.1.txt", "r") as f:
	lines = f.readlines()


positive_words = []
for line in lines:
	if not line.startswith("#"):
		 word, score = line.strip().split("\t")
		 if float(score) > 0:
		 	positive_words.append([word.split("#")[0].lower(), float(score)])


with open("sarcasm_dataset_indices.pickle", "rb") as f:
	split_indices = pickle.load(f)

# bert
# resnet
# ferplus
with open("sarcasm_vat_ferplus.dataset", "rb") as f:
	dataset = pickle.load(f)


with open("sarcasm_last_bert.dataset", "rb") as f:
	bert_words = pickle.load(f)


# global features

with open("bert_global_embedding.pickle", "rb") as f:
	bert_global_embedding = pickle.load(f)


with open("sarcasm_resnet_utterance.pickle", "rb") as f:
	resnet_global_embedding = pickle.load(f)


with open("sarcasm_opensmile_utterance.dataset", "rb") as f:
	opensmile_global_embedding = pickle.load(f)

with open("audio_mask.pickle", "rb") as f:
	_audio_mask = pickle.load(f)

audio_masks = []

for i in range(1, 6):
	audio_masks.append(_audio_mask[i])

audio_masks.append(_audio_mask[0])




keys = list(dataset.keys())



with open("sarcasm_data.json") as f:
	json_data = json.load(f)


train_keys = []
test_keys = []


for key in dataset.keys():
	if json_data[key]["show"] == "FRIENDS":
		test_keys.append(key)
	else:
		train_keys.append(key)
split_indices.append([train_keys, test_keys])



dataset_splits = []

dataset_stats = []

for i in range(6):
	train_keys, test_keys = split_indices[i]
	random.shuffle(train_keys)

	lens = int(len(train_keys) * 0.8)

	train_set = []
	for key in train_keys[:lens]:
		if key in keys:
			train_set.append(key)

	dev_set = []
	for key in train_keys[lens:]:
		if key in keys:
			dev_set.append(key)

	test_set = []
	for key in test_keys:
		if key in keys:
			test_set.append(key)

	all_train_set = []
	for key in train_keys:
		if key in keys:
			all_train_set.append(key)

	dataset_splits.append({"train":train_set, "dev":dev_set, "test":test_set, "all_train":all_train_set})

	dataset_stats.append({"train":cal_dataset_stats(dataset, train_set), "all_train":cal_dataset_stats(dataset, all_train_set)})














class SarcasmDataset(Dataset):
	def __init__(self, dname="train", idx=0, stat_name="train", p_value=0.5):
		self.dataset = dataset
		self.dname = dname
		self.vdims = 35
		self.adims = 65
		self.tdims = 768
		self.dia_names = dataset_splits[idx][self.dname]
		self.state = dataset_stats[idx][stat_name]
		self.audio_mask = audio_masks[idx]
		self.positive_words = [w for w,s in positive_words if s > p_value]

	def __getitem__(self, idx):

		utt_name = self.dia_names[idx]
		print(utt_name)

		feat = self.dataset[utt_name]

		visual_feats = []

		if 	feat["v"] is not None:
			for feats in feat["v"]:
				if feats.shape[0] > 0:
					visual_feats.append((feats-self.state[0])/self.state[1])
				else:
					visual_feats.append(feats)


		acoustic_feats = []

		if 	feat["a"] is not None:
			for feats in feat["a"]:
				if feats.shape[0] > 0:
					acoustic_feats.append((feats-self.state[2])/self.state[3])
				else:
					acoustic_feats.append(feats)

		textual_feats = bert_words[utt_name]["t"]


		bert_global = torch.from_numpy(bert_global_embedding[utt_name]).float()

		# 1000
		audio_global = torch.from_numpy(opensmile_global_embedding[utt_name][:,self.audio_mask]).float()

		# 2048
		if utt_name in resnet_global_embedding.keys():
			video_global = torch.from_numpy(resnet_global_embedding[utt_name]).view(1, -1).float()
		else:
			video_global = torch.zeros((1, 2048))



		mask = [int(w in self.positive_words) for w in feat["words"]]

		label = int(json_data[utt_name]["sarcasm"])

		example = {"visual_feats":visual_feats, "acoustic_feats":acoustic_feats, "textual_feats":textual_feats, 
				"mask":mask, "bert_global":bert_global, "audio_global":audio_global, "video_global":video_global,"label":label }

		return example


	def __len__(self):
		return len(self.dia_names)


def collate_fn(batch):

	vdims = 2048
	adims = 65
	tdims = 768


	visual_nums = []
	acoustic_nums = []

	text_nums = []


	visual_flags = []
	acoustic_flags = []

	bert_global_reps = []
	audio_global_reps = []	
	video_global_reps = []

	for example in batch:

		bert_global_reps.append(example["bert_global"].view(1,-1))
		audio_global_reps.append(example["audio_global"])
		video_global_reps.append(example["video_global"])

		acoustic_nums.append(max([len(v) for v in example["acoustic_feats"]]))

		acoustic_flag = []
		for i in range(len(example["acoustic_feats"])):
			acoustic_flag.append(1)


		text_nums.append(len(example["textual_feats"]))

		visual_feats = example["visual_feats"]
		visual_flag = []

		if len(visual_feats) == 0:
			visual_nums.append(1)
			temp = []
			for i in range(len(example["textual_feats"])):
				temp.append(np.empty((0,vdims)))
				visual_flag.append(0)
			example["visual_feats"] = temp

		
		else:
			visual_nums.append(max([len(v) for v in example["visual_feats"]]))
			for i in range(len(example["visual_feats"])):
				visual_flag.append(1)


		visual_flags.append(visual_flag)
		acoustic_flags.append(acoustic_flag)

	bert_global_reps = torch.cat(bert_global_reps, 0)
	audio_global_reps = torch.cat(audio_global_reps, 0)
	video_global_reps = torch.cat(video_global_reps, 0)

	
	if max(acoustic_nums) == 0:
		acoustic_nums.append(1)

	if max(visual_nums) == 0:
		visual_nums.append(1)


	bsz = len(batch)
	MAX_WORD = min(128, max(text_nums))

	# visual feats speakers
	MAX_LEN = min(100, max(visual_nums))
	video_tensor = torch.zeros((bsz, MAX_WORD, MAX_LEN, vdims))

	for i_batch in range(bsz):
		for i_utt, input_row in enumerate(batch[i_batch]["visual_feats"]):
			video_tensor[i_batch, i_utt, :min(MAX_LEN, len(input_row))] = torch.Tensor(input_row[:min(MAX_LEN, len(input_row)), -vdims:])

	vlens = []
	for i_batch in range(bsz):
		vlen = []
		for input_row in batch[i_batch]["visual_feats"]:
			vlen.append(max(1, min(MAX_LEN, len(input_row))))

		vlen = vlen[:MAX_WORD]
		now_len = len(vlen)
		for ii in range(MAX_WORD-now_len):
			vlen.append(1)

		vlens.append(vlen)

	# acoustic features speakers 
	MAX_LEN = min(300, max(acoustic_nums))
	audio_tensor = torch.zeros((bsz, MAX_WORD, MAX_LEN, adims))

	for i_batch in range(bsz):
		for i_utt, input_row in enumerate(batch[i_batch]["acoustic_feats"]):
			audio_tensor[i_batch, i_utt, :min(MAX_LEN, len(input_row))] = torch.Tensor(input_row[:min(MAX_LEN, len(input_row))])


	alens = []
	for i_batch in range(bsz):
		alen = []
		for input_row in batch[i_batch]["acoustic_feats"]:
			alen.append(max(1, min(MAX_LEN, len(input_row))))

		alen = alen[:MAX_WORD]
		now_len = len(alen)
		for ii in range(MAX_WORD-now_len):
			alen.append(1)

		alens.append(alen)


	# textual features speakers 
	MAX_LEN = MAX_WORD
	text_tensor = torch.zeros((bsz, MAX_WORD, tdims))

	for i_batch in range(bsz):
		for i_utt, input_row in enumerate(batch[i_batch]["textual_feats"]):
			text_tensor[i_batch, i_utt, :min(MAX_LEN, len(input_row))] = torch.Tensor(input_row[:min(MAX_LEN, len(input_row))])

	tlens = []
	for i_batch in range(bsz):
		tlens.append(min(MAX_WORD, len(batch[i_batch]["textual_feats"])))



	masks = []
	for i_batch in range(bsz):

		mask = []
		for m in batch[i_batch]["mask"]:
			mask.append(m)

		mask = mask[:MAX_WORD]
		now_len = len(mask)
		for ii in range(MAX_WORD-now_len):
			mask.append(0)

		masks.append(mask)

	v_masks = []
	for i_batch in range(bsz):
		mask = []
		for m in visual_flags[i_batch]:
			mask.append(m)

		mask = mask[:MAX_WORD]
		now_len = len(mask)
		for ii in range(MAX_WORD-now_len):
			mask.append(0)
		v_masks.append(mask)


	a_masks = []
	for i_batch in range(bsz):
		mask = []
		for m in acoustic_flags[i_batch]:
			mask.append(m)

		mask = mask[:MAX_WORD]
		now_len = len(mask)
		for ii in range(MAX_WORD-now_len):
			mask.append(0)
		a_masks.append(mask)


	labels = []
	for i_batch in range(bsz):
		labels.append(batch[i_batch]["label"])


	labels = torch.Tensor(labels).long()
	masks = torch.Tensor(masks).float()
	v_masks = torch.Tensor(masks).float()
	a_masks = torch.Tensor(masks).float()

	return bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, labels


