# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

import numpy as np 
import torch
import torch.nn as nn
from .Model import Model
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable
from collections import Counter
from . import ConvKB


class OptimizedEnsemble(Model):

	def __init__(self, data_folder, batch_size, ent_tot, rel_tot, neg_examples=0,sampling_mode=None, margin = None, epsilon = None):
		super(OptimizedEnsemble, self).__init__(ent_tot, rel_tot)

		self.batch_size=batch_size
		self.margin = margin
		self.epsilon = epsilon
		self.p_norm = 1
		self.neg_examples=neg_examples
		self.logsigmoid = nn.LogSigmoid()
		self.sig=nn.Sigmoid()
		self.sampling_mode = sampling_mode

		Optimize_Pre_Embed=True

		# Read source embeddings
		TransE_embeddings=np.load("./././PreTrained_KGEs/transe_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		transe_ent_weight = torch.FloatTensor(get_embedding(TransE_embeddings, 'ent_embeddings.weight'))
		transe_rel_weight=torch.FloatTensor(get_embedding(TransE_embeddings,'rel_embeddings.weight'))
		self.transe_ent_embed=nn.Embedding.from_pretrained(transe_ent_weight)
		self.transe_ent_embed.weight.requires_grad=Optimize_Pre_Embed
		self.transe_rel_embed=nn.Embedding.from_pretrained(transe_rel_weight)
		self.transe_rel_embed.weight.requires_grad=Optimize_Pre_Embed

		DistMult_embeddings=np.load("./././PreTrained_KGEs/distmult_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		distmult_ent_weight=torch.FloatTensor(get_embedding(DistMult_embeddings,'ent_embeddings.weight'))
		distmult_rel_weight=torch.FloatTensor(get_embedding(DistMult_embeddings,'rel_embeddings.weight'))
		self.distmult_ent_embed=nn.Embedding.from_pretrained(distmult_ent_weight)
		self.distmult_ent_embed.weight.requires_grad=Optimize_Pre_Embed
		self.distmult_rel_embed=nn.Embedding.from_pretrained(distmult_rel_weight)
		self.distmult_rel_embed.weight.requires_grad=Optimize_Pre_Embed

		RotatE_embeddings=np.load("./././PreTrained_KGEs/rotate_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		rotate_ent_weight=torch.FloatTensor(RotatE_embeddings['ent_embeddings.weight'])
		rotate_rel_weight=torch.FloatTensor(RotatE_embeddings['rel_embeddings.weight'])
		self.rotate_margin=torch.FloatTensor(RotatE_embeddings['margin'])
		self.rotate_margin.requires_grad=False
		self.rotate_ent_embedding_range=torch.FloatTensor(RotatE_embeddings['ent_embedding_range'])
		self.rotate_rel_embedding_range=torch.FloatTensor(RotatE_embeddings['rel_embedding_range'])
		self.rotate_ent_embedding_range.requires_grad=False
		self.rotate_rel_embedding_range.requires_grad=False
		self.rotate_ent_embed=nn.Embedding.from_pretrained(rotate_ent_weight)
		self.rotate_ent_embed.weight.requires_grad=Optimize_Pre_Embed
		self.rotate_rel_embed=nn.Embedding.from_pretrained(rotate_rel_weight)
		self.rotate_rel_embed.weight.requires_grad=Optimize_Pre_Embed

		Complex_embeddings=np.load("./././PreTrained_KGEs/complex_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
		complex_ent_re_weight=torch.FloatTensor(get_embedding(Complex_embeddings,'ent_re_embeddings.weight'))
		complex_rel_re_weight=torch.FloatTensor(get_embedding(Complex_embeddings,'rel_re_embeddings.weight'))
		complex_ent_im_weight=torch.FloatTensor(get_embedding(Complex_embeddings,'ent_im_embeddings.weight'))
		complex_rel_im_weight=torch.FloatTensor(get_embedding(Complex_embeddings,'rel_im_embeddings.weight'))
		self.complex_ent_re_embed=nn.Embedding.from_pretrained(complex_ent_re_weight)
		self.complex_ent_re_embed.weight.requires_grad=Optimize_Pre_Embed
		self.complex_rel_re_embed=nn.Embedding.from_pretrained(complex_rel_re_weight)
		self.complex_rel_re_embed.weight.requires_grad=Optimize_Pre_Embed
		self.complex_ent_im_embed=nn.Embedding.from_pretrained(complex_ent_im_weight)
		self.complex_ent_im_embed.weight.requires_grad=Optimize_Pre_Embed
		self.complex_rel_im_embed=nn.Embedding.from_pretrained(complex_rel_im_weight)
		self.complex_rel_im_embed.weight.requires_grad=Optimize_Pre_Embed

		self.convkb_kernel_size=1
		if data_folder=='FB15K237' or data_folder=='NELL-995':
			self.convkb_out_channels=128
			self.convkb_dim=100
		elif data_folder=='WN18RR':
			self.convkb_out_channels=64
			self.convkb_dim=50

		self.convkb=ConvKB(data_folder = data_folder,ent_tot = self.ent_tot,rel_tot = self.rel_tot ,dim = self.convkb_dim,out_channels=self.convkb_out_channels,use_init_embeddings=True)
		self.convkb.load_state_dict(torch.load('./././PreTrained_KGEs/ConvKB_%s.ckpt'%data_folder))
		self.convkb.eval()

		for name,param in self.convkb.named_parameters():
			param.requires_grad=Optimize_Pre_Embed 

	def _calc_transe(self, h, t, r, mode):

		score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()

		return score

	def _calc_distmult(self, h, t, r, mode):

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

		'''was not here before'''
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

	def _calc_convkb(self, h, t, r,mode):

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


	def forward(self, data, bsize): 

		# forward function for weighting losses not scores
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		rel_id=batch_r[0]
		mode = data['mode']

		score_1=self.compute_transe(batch_h,batch_t,batch_r,mode)
		score_2=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		score_3=self.compute_complex(batch_h,batch_t,batch_r)
		score_4=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		score_5=self.compute_convkb(batch_h,batch_t,batch_r,mode)

	
		return score_1,score_2,score_3,score_4,score_5


	def forward2(self, data): 
		# This is a standard evaluation  
		
		# print(self.weights.weight)
		# print(F.softmax(self.weights.weight,dim=1))

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		rel_id=batch_r[0]
		mode = data['mode']

		# we can make this part in Tester.py and use the same forward for train and predict(test)
		if mode=='head_batch':
			batch_t=torch.cat([batch_t]*self.ent_tot)
			batch_r=torch.cat([batch_r]*self.ent_tot)
		elif mode=='tail_batch':
			batch_h=torch.cat([batch_h]*self.ent_tot)
			batch_r=torch.cat([batch_r]*self.ent_tot)


		score_1 =self.compute_transe(batch_h,batch_t,batch_r,mode)
		score_1=(score_1-score_1.min())/(score_1.max()-score_1.min())
		score_1 =  1 - score_1


		score_2=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		score_2=(score_2-score_2.min())/(score_2.max()-score_2.min())
		
		score_3=self.compute_complex(batch_h,batch_t,batch_r)
		score_3=(score_3-score_3.min())/(score_3.max()-score_3.min())


		score_4=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		score_4=(score_4-score_4.min())/(score_4.max()-score_4.min())


		score_5=self.compute_convkb(batch_h,batch_t,batch_r,mode)
		score_5=(score_5-score_5.min())/(score_5.max()-score_5.min())

		
		avg_score=  (score_1+score_2+score_3+score_4+score_5)/5.0
		return avg_score



	def forward2_PoEs(self, data): 
		# This is an evaluation for a Committee-Based Knowledge Graph Embedding Paper (Product of Experts PoEs)
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		# we can make this part in Tester.py and use the same forward for train and predict(test)
		if mode=='head_batch':
			batch_t=torch.cat([batch_t]*self.ent_tot)
			batch_r=torch.cat([batch_r]*self.ent_tot)
		elif mode=='tail_batch':
			batch_h=torch.cat([batch_h]*self.ent_tot)
			batch_r=torch.cat([batch_r]*self.ent_tot)


		score_1=self.compute_transe(batch_h,batch_t,batch_r,mode).view(1,-1)
		score_1 = torch.sigmoid(-score_1)
		score_2=self.compute_distmult(batch_h,batch_t,batch_r,mode).view(1,-1)
		score_2 = torch.sigmoid(score_2)
		score_3=self.compute_complex(batch_h,batch_t,batch_r).view(1,-1)
		score_3 = torch.sigmoid(score_3)
		score_4=self.compute_rotate(batch_h,batch_t,batch_r,mode).view(1,-1)
		score_4 = torch.sigmoid(score_4)
		score_5=self.compute_convkb(batch_h,batch_t,batch_r,mode).view(1,-1)
		score_5 = torch.sigmoid(score_5)

		score = score_1*score_2*score_3*score_4*score_5
		score = score / score.sum()

		return score

	def forward3(self, data): 

		# print(F.softmax(self.weights.weight,dim=1))
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		
		rel_weights = self.weights(batch_r)
		best=torch.argmax(rel_weights)
		

		if best==0:
			score=self.compute_transe(batch_h,batch_t,batch_r,mode)
		elif best==1:
			score=self.compute_distmult(batch_h,batch_t,batch_r,mode)
		elif best==2:
			score=self.compute_complex(batch_h,batch_t,batch_r)
		elif best==3:
			score=self.compute_rotate(batch_h,batch_t,batch_r,mode)
		elif best==4: 
			score=self.compute_convkb(batch_h,batch_t,batch_r,mode)

		return score

	def compute_alpha(self,score1,score2,weight1,weight2):
		score1=torch.mul(weight1,score1)
		p_score1 = score1[:self.batch_size]
		p_score1 = p_score1.view(-1, self.batch_size).permute(1, 0)

		n_score1 = score1[self.batch_size:]
		n_score1 = n_score1.view(-1, self.batch_size).permute(1, 0)

		diff1=(p_score1 - n_score1).flatten()
		m_1=torch.mean(diff1)
		var_1=torch.var(diff1)

		score2=torch.mul(weight2,score2)
		p_score2 = score2[:self.batch_size]
		p_score2 = p_score2.view(-1, self.batch_size).permute(1, 0)

		n_score2 = score2[self.batch_size:]
		n_score2 = n_score2.view(-1, self.batch_size).permute(1, 0)

		diff2=(p_score2 - n_score2).flatten()
		m_2=torch.mean(diff2)
		var_2=torch.var(diff2)

		alpha=(var_1*m_2)/(var_2*m_1)
		return alpha
	
	def compute_alpha_ri(self,data,score, bsize):
		mean_=np.zeros(5)
		var_=np.zeros(5)
		alpha_=np.zeros(5)
		batch_r = data['batch_r']
		rel=batch_r[0]
		for i in range(5):
			p_score = score[:,i][:bsize]
			p_score = p_score.view(-1, bsize).permute(1, 0)
			n_score = score[:,i][bsize:]
			n_score = n_score.view(-1, bsize).permute(1, 0)

			res=(p_score-n_score).flatten()
			res=res.cpu().data.numpy()

			mean_[i]=np.mean(res)
			var_[i]=np.var(res)
			alpha_[i]=(var_[0]*mean_[i])/(var_[i]*mean_[0])

		self.alphas.weight.data[rel] = torch.from_numpy(alpha_).cuda()

	def compute_transe(self,batch_h,batch_t,batch_r,mode):
		
		h = self.transe_ent_embed(batch_h)
		t = self.transe_ent_embed(batch_t)
		r = self.transe_rel_embed(batch_r)
		h = F.normalize(h , 2, -1)
		t = F.normalize(t , 2, -1)
		r = F.normalize(r, 2, -1)
		score = self._calc_transe(h ,t, r, mode)

		return score

	def compute_distmult(self,batch_h,batch_t,batch_r,mode):

		h = self.distmult_ent_embed(batch_h)
		t = self.distmult_ent_embed(batch_t)
		r = self.distmult_rel_embed(batch_r)

		score = self._calc_distmult(h ,t, r, mode)

		return score

	def compute_rotate(self,batch_h,batch_t,batch_r,mode):

		h = self.rotate_ent_embed(batch_h)
		t = self.rotate_ent_embed(batch_t)
		r = self.rotate_rel_embed(batch_r)
		score = self.rotate_margin.item() - self._calc_rotate(h ,t, r, mode)
		return score

	def compute_complex(self,batch_h,batch_t,batch_r):
		h_re = self.complex_ent_re_embed(batch_h)
		h_im = self.complex_ent_im_embed(batch_h)
		t_re = self.complex_ent_re_embed(batch_t)
		t_im = self.complex_ent_im_embed(batch_t)
		r_re = self.complex_rel_re_embed(batch_r)
		r_im = self.complex_rel_im_embed(batch_r)
		score = self._calc_complex(h_re, h_im, t_re, t_im, r_re, r_im)
		return score

	def compute_convkb(self,batch_h,batch_t,batch_r,mode):
	
		h = self.convkb.ent_embeddings(batch_h)
		r = self.convkb.rel_embeddings(batch_r)
		t = self.convkb.ent_embeddings(batch_t)

		'''before, was useful for testing only'''

		score = self._calc_convkb(h,t,r,mode)
		return score


	def predict(self, data):
		score =  - self.forward2(data)
		return score.cpu().data.numpy()
		
def get_embedding(embeddings, key):
	# Try string key first
	if key in embeddings:
		return embeddings[key]

	# Try bytes key
	key_bytes = key.encode()
	if key_bytes in embeddings:
		return embeddings[key_bytes]

	raise KeyError(f"Key '{key}' not found in embeddings.")
