import numpy as np
seed = 42
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils




class TAttention_Mask_Relu_Score(nn.Module):
	def __init__(self):
		super(TAttention_Mask_Relu_Score, self).__init__()
		config = {"input_dims":[768, 65, 2048], "global_input_dims":[768, 1000, 2048],"hidden_dims": 200, "output_dims": 2, "dpout":0.5}
		dt, da, dv = config["input_dims"]
		gdt, gda, gdv = config["global_input_dims"]
		h = config["hidden_dims"]
		output_dim = config["output_dims"]
		dropout = config["dpout"]

		self.linear_t = nn.Linear(dt, h)
		self.linear_v = nn.Linear(dv, h)
		self.linear_a = nn.Linear(da, h)

		self.importance_ta = nn.Linear(h+h, 1)
		self.importance_tv = nn.Linear(h+h, 1)
		self.importance_t = nn.Linear(h, 1)

		self.fc = nn.Linear(gdt+gda+gdv+dt+da+dv, output_dim)
		self.dropout = nn.Dropout(dropout)
		self.h = h


	def forward(self, bert_global_reps, audio_global_reps, video_global_reps, video_tensor, vlens, audio_tensor, alens, text_tensor, tlens, masks, v_masks, a_masks, get_feature=False):
		
		batch_size, word_num,  _ = text_tensor.size()


		vl = video_tensor.size(2)
		al = audio_tensor.size(2)

		global_h1 = torch.zeros(1, batch_size, self.h)
		if torch.cuda.is_available():
			global_h1 = global_h1.cuda()


		vvlens = []
		for vlen in vlens:
			for l in vlen:
				vvlens.append(l)

		vvlens_tensor = torch.Tensor(vvlens)
		if torch.cuda.is_available():
			vvlens_tensor = vvlens_tensor.cuda()

		video_tensor = video_tensor.contiguous().view(batch_size*word_num, vl, -1)
		video_tensor = torch.sum(video_tensor, 1)/vvlens_tensor.view(batch_size*word_num, 1).expand(batch_size*word_num, video_tensor.size(-1))
		v_last_hs = F.tanh(self.linear_v(video_tensor))
		v_last_hs = v_last_hs.contiguous().view(batch_size, word_num, -1)


		aalens = []
		for alen in alens:
			for l in alen:
				aalens.append(l)


		aalens_tensor = torch.Tensor(aalens)
		if torch.cuda.is_available():
			aalens_tensor = aalens_tensor.cuda()

		audio_tensor = audio_tensor.contiguous().view(batch_size*word_num, al, -1)
		audio_tensor = torch.sum(audio_tensor, 1)/aalens_tensor.view(batch_size*word_num, 1).expand(batch_size*word_num, audio_tensor.size(-1))
		a_last_hs = F.tanh(self.linear_a(audio_tensor))
		a_last_hs = a_last_hs.contiguous().view(batch_size, word_num, -1)


		out_t = F.tanh(self.linear_t(text_tensor))

		output_tensor = torch.cat([text_tensor, audio_tensor.view(batch_size, word_num, -1), video_tensor.view(batch_size, word_num, -1)], -1)


		importance_tv = F.relu(self.importance_tv(torch.cat([out_t, v_last_hs], -1)))
		importance_ta = F.relu(self.importance_ta(torch.cat([out_t, a_last_hs], -1)))

		# masking
		index = torch.arange(word_num).view(1,-1).expand(batch_size, word_num)
		tlens_tensor = torch.Tensor(tlens).unsqueeze(1).expand(batch_size, word_num)
		if torch.cuda.is_available():
			tlens_tensor = tlens_tensor.cuda()
			index = index.cuda()

		importance_tv = importance_tv.squeeze(2) * v_masks * masks

		importance_ta = importance_ta.squeeze(2) * a_masks * masks

		importance_t = self.importance_t(out_t)
		importance_t = importance_t.squeeze(2)

		importances =  importance_t + importance_tv + importance_ta + (1-(index < tlens_tensor).float()) * (-1e8)

		# print(importances)

		global_weights = F.softmax(importances, 1)

		last_hs = torch.sum(global_weights.unsqueeze(2).expand_as(output_tensor) * output_tensor, 1)


		global_inputs = torch.cat([bert_global_reps, audio_global_reps, video_global_reps], -1)

		last_hs = torch.cat([last_hs, global_inputs], -1)

		last_hs = F.tanh(last_hs)

		if get_feature:
			return last_hs

		last_hs = self.dropout(last_hs)
		output = self.fc(last_hs)
		return output










def get_model(name):
	if name == "TAttention_Mask_Relu_Score":
		return TAttention_Mask_Relu_Score()
	