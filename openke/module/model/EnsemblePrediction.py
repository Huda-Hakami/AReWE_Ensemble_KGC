# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

import numpy as np 
import torch
import torch.nn as nn
from .Model import Model
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable
from . import ConvKB
import sys
np.set_printoptions(threshold=sys.maxsize)

class EnsemblePrediction(Model):

	def __init__(self, data_folder, batch_size, ent_tot, rel_tot):
		super(EnsemblePrediction, self).__init__(ent_tot, rel_tot)

		self.batch_size=batch_size
		self.p_norm = 1

		# Read source embeddings
		TransE_embeddings=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/transe_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		transe_ent_weight=torch.FloatTensor(TransE_embeddings['ent_embeddings.weight'])
		transe_rel_weight=torch.FloatTensor(TransE_embeddings['rel_embeddings.weight'])
		self.transe_ent_embed=nn.Embedding.from_pretrained(transe_ent_weight)
		self.transe_ent_embed.weight.requires_grad=False
		self.transe_rel_embed=nn.Embedding.from_pretrained(transe_rel_weight)
		self.transe_rel_embed.weight.requires_grad=False

		DistMult_embeddings=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/distmult_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		distmult_ent_weight=torch.FloatTensor(DistMult_embeddings['ent_embeddings.weight'])
		distmult_rel_weight=torch.FloatTensor(DistMult_embeddings['rel_embeddings.weight'])
		self.distmult_ent_embed=nn.Embedding.from_pretrained(distmult_ent_weight)
		self.distmult_ent_embed.weight.requires_grad=False
		self.distmult_rel_embed=nn.Embedding.from_pretrained(distmult_rel_weight)
		self.distmult_rel_embed.weight.requires_grad=False


		RotatE_embeddings=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/rotate_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		rotate_ent_weight=torch.FloatTensor(RotatE_embeddings['ent_embeddings.weight'])
		rotate_rel_weight=torch.FloatTensor(RotatE_embeddings['rel_embeddings.weight'])
		self.rotate_margin=torch.FloatTensor(RotatE_embeddings['margin'])
		self.rotate_margin.requires_grad=False
		self.rotate_ent_embedding_range=torch.FloatTensor(RotatE_embeddings['ent_embedding_range'])
		self.rotate_rel_embedding_range=torch.FloatTensor(RotatE_embeddings['rel_embedding_range'])
		self.rotate_ent_embedding_range.requires_grad=False
		self.rotate_rel_embedding_range.requires_grad=False
		self.rotate_ent_embed=nn.Embedding.from_pretrained(rotate_ent_weight)
		self.rotate_ent_embed.weight.requires_grad=False
		self.rotate_rel_embed=nn.Embedding.from_pretrained(rotate_rel_weight)
		self.rotate_rel_embed.weight.requires_grad=False

		Complex_embeddings=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/complex_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		complex_ent_re_weight=torch.FloatTensor(Complex_embeddings['ent_re_embeddings.weight'])
		complex_rel_re_weight=torch.FloatTensor(Complex_embeddings['rel_re_embeddings.weight'])
		complex_ent_im_weight=torch.FloatTensor(Complex_embeddings['ent_im_embeddings.weight'])
		complex_rel_im_weight=torch.FloatTensor(Complex_embeddings['rel_im_embeddings.weight'])
		self.complex_ent_re_embed=nn.Embedding.from_pretrained(complex_ent_re_weight)
		self.complex_ent_re_embed.weight.requires_grad=False
		self.complex_rel_re_embed=nn.Embedding.from_pretrained(complex_rel_re_weight)
		self.complex_rel_re_embed.weight.requires_grad=False
		self.complex_ent_im_embed=nn.Embedding.from_pretrained(complex_ent_im_weight)
		self.complex_ent_im_embed.weight.requires_grad=False
		self.complex_rel_im_embed=nn.Embedding.from_pretrained(complex_rel_im_weight)
		self.complex_rel_im_embed.weight.requires_grad=False

		self.convkb_kernel_size=1
		if data_folder=='FB15K237' or data_folder=='NELL-995':
			self.convkb_out_channels=128
			self.convkb_dim=100
		elif data_folder=='WN18RR':
			self.convkb_out_channels=64
			self.convkb_dim=50

		self.convkb=ConvKB(data_folder = data_folder,ent_tot = self.ent_tot,rel_tot = self.rel_tot ,dim = self.convkb_dim,out_channels=self.convkb_out_channels,use_init_embeddings=True)
		self.convkb.load_state_dict(torch.load('/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/ConvKB_%s.ckpt'%data_folder))
		self.convkb.eval()

		for name,param in self.convkb.named_parameters():
			param.requires_grad=False 


		# Read Pre-specified weights
		self.weights_l = nn.Embedding(self.rel_tot, 5)
		# PreRelWeightsPerKGE_%s_mrr_valid.npy
		# alpha_ri_%s.npy
		pre_weights=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreWeights/PreRelWeightsPerKGE_%s_mrr_l_valid.npy"%data_folder)
		# pre_weights=pre_weights/pre_weights.sum(axis=1,keepdims=True)
		self.weights_l.weight.data = torch.from_numpy(pre_weights)
		self.weights_l.weight.requires_grad = False

		self.weights_r = nn.Embedding(self.rel_tot, 5)
		pre_weights=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreWeights/PreRelWeightsPerKGE_%s_mrr_r_valid.npy"%data_folder)
		# pre_weights=pre_weights/pre_weights.sum(axis=1,keepdims=True)
		self.weights_r.weight.data = torch.from_numpy(pre_weights)
		self.weights_r.weight.requires_grad = False

		self.weights = nn.Embedding(self.rel_tot, 5)
		pre_weights=pre_weights/pre_weights.sum(axis=1,keepdims=True)
		pre_weights=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreWeights/PreRelWeightsPerKGE_%s_mrr_valid.npy"%data_folder)
		self.weights.weight.data = torch.from_numpy(pre_weights)
		self.weights.weight.requires_grad = False

		self.alphas = nn.Embedding(self.rel_tot, 5)
		alphas=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/alpha_ri_%s.npy"%data_folder)
		self.alphas.weight.data = torch.from_numpy(alphas)
		self.alphas.weight.requires_grad = False

		
		# print(self.alphas.weight)
		# exit()
		# print(F.softmax(self.weights.weight,dim=1))
		# exit()
		
	def _calc_transe(self, h, t, r, mode):

		if mode == 'head_batch':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = h + (r - t)

		elif mode=='tail_batch':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = (h + r) - t

		elif mode=='rel_batch':
			h = h.view(-1, h.shape[0], h.shape[-1])
			t = t.view(-1, t.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = (h + r) - t
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()

		return score

	def _calc_distmult(self, h, t, r, mode):
		if mode == 'head_batch':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = h * (r * t)

		elif mode=='tail_batch':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = (h * r) * t
	
		elif mode=='rel_batch':
			h = h.view(-1, h.shape[0], h.shape[-1])
			t = t.view(-1, t.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = (h * r) * t	 
		else:
			score = (h * r) * t
		score = torch.sum(score, -1).flatten()

		return score

	def _calc_complex(self, h_re, h_im, t_re, t_im, r_re, r_im):
		return torch.sum(h_re * t_re * r_re+ h_im * t_im * r_re + h_re * t_im * r_im- h_im * t_re * r_im,-1).flatten()

	def _calc_rotate(self, h, t, r, mode):
		pi = 3.14159265358979323846

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)

		phase_relation = r / (self.rotate_rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)
		
		if mode == "head_batch":
			re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
			re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
			im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
			im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
			im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
			re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head

		elif mode=="rel_batch":
			re_head = re_head.view(-1, re_head.shape[0], re_head.shape[-1]).permute(1, 0, 2)
			re_tail = re_tail.view(-1, re_tail.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
			im_head = im_head.view(-1, im_head.shape[0], im_head.shape[-1]).permute(1, 0, 2)
			im_tail = im_tail.view(-1, im_tail.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
			im_relation = im_relation.view(-1, re_relation.shape[0], re_relation.shape[-1])
			re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1])
			

			re_score = re_head * re_tail + im_head * im_tail
			im_score = re_head * im_tail - im_head * re_tail
			re_score = re_score - re_relation
			im_score = im_score - im_relation
		
		else:
			re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
			re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
			im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
			im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
			im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
			re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		return score.permute(1, 0).flatten()

	def _calc_convkb(self, h, t, r):
		h = h.unsqueeze(1) # bs x 1 x dim
		r = r.unsqueeze(1)
		t = t.unsqueeze(1)

		conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
		conv_input = conv_input.transpose(1, 2)
		# To make tensor of size 4, where second dim is for input channels
		conv_input = conv_input.unsqueeze(1)
		conv_input = self.convkb.conv1_bn(conv_input)
		out_conv = self.convkb.conv_layer(conv_input)
		out_conv = self.convkb.conv2_bn(out_conv)
		out_conv = torch.relu(out_conv)
		out_conv = out_conv.view(-1, (self.convkb_dim - self.convkb_kernel_size + 1) * self.convkb_out_channels)
		input_fc = self.convkb.dropout(out_conv)
		score = self.convkb.fc_layer(input_fc).view(-1)

		return score

	def compute_transe(self,batch_h,batch_t,batch_r,mode):
		
		h = self.transe_ent_embed(batch_h)
		t = self.transe_ent_embed(batch_t)
		r = self.transe_rel_embed(batch_r)
		h = F.normalize(h , 2, -1)
		t = F.normalize(t , 2, -1)
		r = F.normalize(r, 2, -1)
		score = self._calc_transe(h ,t, r, mode)

		min_=score.min()
		max_=score.max()
		score=(score-min_)/(max_-min_)
		score=1-score
		# score=torch.exp(- score)
		# score=torch.sigmoid(-score)
		return score

	def compute_distmult(self,batch_h,batch_t,batch_r,mode):

		h = self.distmult_ent_embed(batch_h)
		t = self.distmult_ent_embed(batch_t)
		r = self.distmult_rel_embed(batch_r)

		score = self._calc_distmult(h ,t, r, mode)
		min_=score.min()
		max_=score.max()
		score=(score-min_)/(max_-min_)
		# score=(score-score.mean())/(score.std())
		# score=torch.exp(score)
		# score=torch.sigmoid(score)
		return score

	def compute_rotate(self,batch_h,batch_t,batch_r,mode):

		h = self.rotate_ent_embed(batch_h)
		t = self.rotate_ent_embed(batch_t)
		r = self.rotate_rel_embed(batch_r)
		score = self.rotate_margin.item() - self._calc_rotate(h ,t, r, mode)
		min_=score.min()
		max_=score.max()
		score=(score-min_)/(max_-min_)
		# score=torch.exp(score)
		# score=torch.sigmoid(score)
		return score

	def compute_complex(self,batch_h,batch_t,batch_r):
		h_re = self.complex_ent_re_embed(batch_h)
		h_im = self.complex_ent_im_embed(batch_h)
		t_re = self.complex_ent_re_embed(batch_t)
		t_im = self.complex_ent_im_embed(batch_t)
		r_re = self.complex_rel_re_embed(batch_r)
		r_im = self.complex_rel_im_embed(batch_r)
		score = self._calc_complex(h_re, h_im, t_re, t_im, r_re, r_im)
		min_=score.min()
		max_=score.max()
		score=(score-min_)/(max_-min_)
		# score=torch.exp(score)
		# score=torch.sigmoid(score)
		return score

	def compute_convkb(self,batch_h,batch_t,batch_r,mode):
		h = self.convkb.ent_embeddings(batch_h)
		r = self.convkb.rel_embeddings(batch_r)
		t = self.convkb.ent_embeddings(batch_t)
		if mode == 'head_batch':
			r=torch.cat([r]*h.shape[0])
			t=torch.cat([t]*h.shape[0])
		elif mode=='tail_batch':
			h=torch.cat([h]*t.shape[0])
			r=torch.cat([r]*t.shape[0])
		elif mode=='rel_batch':
			h=torch.cat([h]*r.shape[0])
			t=torch.cat([t]*r.shape[0])
		score = self._calc_convkb(h,t,r)
		min_=score.min()
		max_=score.max()
		score=(score-min_)/(max_-min_)
		# score=torch.exp(score)
		# score=torch.sigmoid(score)
		return score

	def Avg_Baseline(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		rel_weights_l = self.weights_l(batch_r)
		rel_weights_r = self.weights_r(batch_r)
		rel_weights = self.weights(batch_r)
		alphas = self.alphas(batch_r)
		rel_weights_l=F.softmax(rel_weights_l,dim=1)
		rel_weights_r=F.softmax(rel_weights_r,dim=1)
		rel_weights=F.softmax(rel_weights,dim=1)


		# score1=self.compute_transe(batch_h,batch_t,batch_r,mode)
		# score2=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		# score3=self.compute_complex(batch_h,batch_t,batch_r)
		score4=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		# score5=self.compute_convkb(batch_h,batch_t,batch_r,mode)

		# print(score_1[:3])
		# print(score_2[:3])
		# print(score_3[:3])
		# print(score_4[:3])
		# print(score_5[:3])
		
		# al=[1.0,0.348,0.424,1.289,0.825]
		# score=(al[0]*score1)+(al[1]*score2)+(al[2]*score3)+(al[3]*score4)+(al[4]*score5)
		# score=(rel_weights_r[0,0]*score1)+(rel_weights_r[0,1]*score2)+(rel_weights_r[0,2]*score3)+(rel_weights_r[0,3]*score4)+(rel_weights_r[0,4]*score5)
		# score=(alphas[0,0]*score1)+(alphas[0,1]*score2)+(alphas[0,2]*score3)+(alphas[0,3]*score4)+(alphas[0,4]*score5)
		# if mode=='head_batch':	
		# 	score=(rel_weights_l[0,0]*score1)+(rel_weights_l[0,1]*score2)+(rel_weights_l[0,2]*score3)+(rel_weights_l[0,3]*score4)+(rel_weights_l[0,4]*score5)

		# elif mode=='tail_batch':
		# 	score=(rel_weights_r[0,0]*score1)+(rel_weights_r[0,1]*score2)+(rel_weights_r[0,2]*score3)+(rel_weights_r[0,3]*score4)+(rel_weights_r[0,4]*score5)

		# heads=[.227,.395,.377,.463,.237]
		# heads=heads/np.sum(heads)
		# tails=[.27,.449,.444,.521,.286]
		# tails=tails/np.sum(tails)
		# if mode=='head_batch':
		# 	score = ((heads[0]*score1)+(heads[1]*score2)+(heads[2]*score3)+(heads[3]*score4)+(heads[4]*score5))
		# elif mode=='tail_batch':
		# 	score = ((tails[0]*score1)+(tails[1]*score2)+(tails[2]*score3)+(tails[3]*score4)+(tails[4]*score5))

		# score1 = torch.sigmoid(-score1)
		# score2=torch.sigmoid(score2)
		# score3=torch.sigmoid(score3)
		# score4=torch.sigmoid(score4)
		# score5=torch.sigmoid(score5)
		# score = score1*score2*score3*score4*score5
		# score = score / score.sum()

		# score=(score1+score2+score3+score4+score5)
		# score = ((rel_weights[0,0]*score1)+(rel_weights[0,1]*score2)+(rel_weights[0,2]*score3)+(rel_weights[0,3]*score4)+(rel_weights[0,4]*score5))
		return score4

	def Avg_Baseline_using_ranks(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']


		t=torch.tensor([i for i in range(self.ent_tot)]).cuda()

		rel_weights_l = self.weights_l(batch_r)
		rel_weights_r = self.weights_r(batch_r)
		rel_weights = self.weights(batch_r)
		alphas = self.alphas(batch_r)
		# rel_weights=F.softmax(rel_weights,dim=1)


		score_1=self.compute_transe(batch_h,batch_t,batch_r,mode)
		score_1_ranks=torch.argsort(-score_1)
		ranks1=torch.where(score_1_ranks==t[:,None])[1]+1
		ranks1=ranks1.to(torch.float)
		
		score_2=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		score_2_ranks=torch.argsort(-score_2)
		ranks2=torch.where(score_2_ranks==t[:,None])[1]+1
		ranks2=ranks2.to(torch.float)

		score_3=self.compute_complex(batch_h,batch_t,batch_r)
		score_3_ranks=torch.argsort(-score_3)
		ranks3=torch.where(score_3_ranks==t[:,None])[1]+1
		ranks3=ranks3.to(torch.float)

		score_4=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		score_4_ranks=torch.argsort(-score_4)
		ranks4=torch.where(score_4_ranks==t[:,None])[1]+1
		ranks4=ranks4.to(torch.float)

		score_5=self.compute_convkb(batch_h,batch_t,batch_r,mode)
		score_5_ranks=torch.argsort(-score_5)
		ranks5=torch.where(score_5_ranks==t[:,None])[1]+1
		ranks5=ranks5.to(torch.float)

		# print(score_1[:3])
		# print(score_2[:3])
		# print(score_3[:3])
		# print(score_4[:3])
		# print(score_5[:3])
		

		# score=score_1+score_2+score_4
		# score=(1.0*score_1)+(1.0*score_2)+(1.0*score_3)+(1.0*score_4)+(1.0*score_5)
		# score=(ranks1+ranks2+ranks3+ranks4+ranks5)
		# score = ((rel_weights[0,0]*ranks1)+(rel_weights[0,1]*ranks2)+(rel_weights[0,2]*ranks3)+
		# 		 (rel_weights[0,3]*ranks4)+(rel_weights[0,4]*ranks5))

		if mode=='head_batch':	
			score=(rel_weights_l[0,0]*ranks1)+(rel_weights_l[0,1]*ranks2)+(rel_weights_l[0,2]*ranks3)+(rel_weights_l[0,3]*ranks4)+(rel_weights_l[0,4]*ranks5)

		elif mode=='tail_batch':
			score=(rel_weights_r[0,0]*ranks1)+(rel_weights_r[0,1]*ranks2)+(rel_weights_r[0,2]*ranks3)+(rel_weights_r[0,3]*ranks4)+(rel_weights_r[0,4]*ranks5)


		# score=(rel_weights[0,0]*ranks1)+(rel_weights[0,1]*ranks2)+(rel_weights[0,2]*ranks3)+(rel_weights[0,3]*ranks4)+(rel_weights[0,4]*ranks5)

		return score

	def Avg_Baseline_Top1(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		if mode == 'head_batch':
			rel_weights = self.weights_l(batch_r)
		elif mode == 'tail_batch':
			rel_weights = self.weights_r(batch_r)
		# rel_weights = self.weights(batch_r)
		best=torch.argmax(rel_weights)
		

		if best==0:
			# print("transe best")
			score=self.compute_transe(batch_h,batch_t,batch_r,mode)
		elif best==1:
			# print("distmult best")
			score=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		elif best==2:
			# print("complex best")
			score=self.compute_complex(batch_h,batch_t,batch_r)
		elif best==3:
			# print("rotate best")
			score=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		elif best==4: 
			# print("convkb best")
			score=self.compute_convkb(batch_h,batch_t,batch_r,mode)

		return score

	def Avg_Baseline_Top2(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		if mode == 'head_batch':
			rel_weights = self.weights_l(batch_r)
		elif mode == 'tail_batch':
			rel_weights = self.weights_r(batch_r)
		best=torch.argsort(-rel_weights).flatten()

		
		
		if (best[0]==0 and best[1]==1) or (best[0]==1 and best[1]==0):
			score1=self.compute_transe(batch_h,batch_t,batch_r,mode)
			score2=self.compute_distmult(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,0]*score1)+(rel_weights[0,1]*score2)
		elif (best[0]==0 and best[1]==2) or (best[0]==2 and best[1]==0):
			score1=self.compute_transe(batch_h,batch_t,batch_r,mode)
			score2=self.compute_complex(batch_h,batch_t,batch_r)
			score=(rel_weights[0,0]*score1)+(rel_weights[0,2]*score2)
		elif (best[0]==0 and best[1]==3) or (best[0]==3 and best[1]==0):
			score1=self.compute_transe(batch_h,batch_t,batch_r,mode)
			score2=self.compute_rotate(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,0]*score1)+(rel_weights[0,3]*score2)
		elif (best[0]==0 and best[1]==4) or (best[0]==4 and best[1]==0):
			score1=self.compute_transe(batch_h,batch_t,batch_r,mode)
			score2=self.compute_convkb(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,0]*score1)+(rel_weights[0,4]*score2)
		elif (best[0]==1 and best[1]==2) or (best[0]==2 and best[1]==1):
			score1=self.compute_distmult(batch_h,batch_t,batch_r,mode)
			score2=self.compute_complex(batch_h,batch_t,batch_r)
			score=(rel_weights[0,1]*score1)+(rel_weights[0,2]*score2)
		elif (best[0]==1 and best[1]==3) or (best[0]==3 and best[1]==1):
			score1=self.compute_distmult(batch_h,batch_t,batch_r,mode)
			score2=self.compute_rotate(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,1]*score1)+(rel_weights[0,3]*score2)
		elif (best[0]==1 and best[1]==4) or (best[0]==4 and best[1]==1):
			score1=self.compute_distmult(batch_h,batch_t,batch_r,mode)
			score2=self.compute_convkb(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,1]*score1)+(rel_weights[0,4]*score2)
		elif (best[0]==2 and best[1]==3) or (best[0]==3 and best[1]==2):
			score1=self.compute_complex(batch_h,batch_t,batch_r)
			score2=self.compute_rotate(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,2]*score1)+(rel_weights[0,3]*score2)
		elif (best[0]==2 and best[1]==4) or (best[0]==4 and best[1]==2):
			score1=self.compute_complex(batch_h,batch_t,batch_r)
			score2=self.compute_convkb(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,2]*score1)+(rel_weights[0,4]*score2)
		elif (best[0]==3 and best[1]==4) or (best[0]==4 and best[1]==3):
			score1=self.compute_rotate(batch_h,batch_t,batch_r,mode)
			score2=self.compute_convkb(batch_h,batch_t,batch_r,mode)
			score=(rel_weights[0,3]*score1)+(rel_weights[0,4]*score2)

		# score=score1+score2

		return score

	def TopK_Normalising(self,data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']


		score1=self.compute_transe(batch_h,batch_t,batch_r,mode)
		score1 = self.get_topk_nomr(score1)

		score2=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		score2 = self.get_topk_nomr(score2)

		score3=self.compute_complex(batch_h,batch_t,batch_r)
		score3 = self.get_topk_nomr(score3)

		score4=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		score4 = self.get_topk_nomr(score4)

		score5=self.compute_convkb(batch_h,batch_t,batch_r,mode)
		score5 = self.get_topk_nomr(score5)

		score= score1 + score2 + score3+ score4+ score5

		return score

	def get_topk_nomr(self,score):
	# This method is implemnting normalizing over top k predictions of each model
# 		From "Ensembles of knowledge graph embedding models
# 		improve predictions for drug discovery"
		# e=torch.tensor([i for i in range(self.ent_tot)]).cuda()
		K=40943
		argsort=torch.argsort(score,descending=True)
		score_topk=score[argsort[:K]]
		min_=score_topk.min()
		score=(score-min_)/(score_topk.max()-min_)
		score_updated=torch.FloatTensor(np.full((score.shape[0]),0.0)).cuda()
		score_updated[argsort[:K]]=score[argsort[:K]]
		# condition = torch.tensor([(x in argsort[:K])  for x in e]).cuda()
		# score=score.where(condition, torch.tensor(0.0))
		return score_updated

	def predict(self, data):
		score =   - self.Avg_Baseline(data)
		return score.cpu().data.numpy()

