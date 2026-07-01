import numpy as np 
import torch
import torch.nn as nn
from .Model import Model
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class FeatureRichNet(Model):

	def __init__(self, data_folder, batch_size, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None):
		super(FeatureRichNet, self).__init__(ent_tot, rel_tot)

		self.batch_size=batch_size
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.p_norm = 1

		# Read source embeddings
		# file_to_read = open("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/transe_embeddings_FB15K237.pickle", "rb")
		# TransE_embeddings_dict = pickle.load(file_to_read)

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, 100)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
		
		#scoring-function
		self.fc1=nn.Linear(700,self.dim)
		self.relu1=nn.ReLU()
		self.fc2=nn.Linear(self.dim,self.dim)
		self.relu2=nn.ReLU()
		self.fc3=nn.Linear(self.dim,1)
		# self.sig = nn.Sigmoid()


	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)


		input_=torch.cat((h,t,r),1)
		out=self.fc1(input_)
		out=self.relu1(out)
		out=self.fc2(out)
		out=self.relu2(out)
		score=self.fc3(out)
		# score = self.sig(out)

		return score.flatten()

	def forward2(self, data):


		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		if mode == 'head_batch':
			r=torch.cat([r]*h.shape[0])
			t=torch.cat([t]*h.shape[0])
		elif mode=='tail_batch':
			h=torch.cat([h]*t.shape[0])
			r=torch.cat([r]*t.shape[0])
		elif mode=='rel_batch':
			h=torch.cat([h]*r.shape[0])
			t=torch.cat([t]*r.shape[0])

		input_=torch.cat((h,t,r),1)
		out=self.fc1(input_)
		out=self.relu1(out)
		out=self.fc2(out)
		out=self.relu2(out)
		score=self.fc3(out)
		# score = self.sig(out)

		return score

	def regularization(self, data):

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		# regul = torch.mean(h ** 2 + t ** 2 + r ** 2)
		return regul


	def l3_regularization(self):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		score = -self.forward2(data)
		return score.cpu().data.numpy()

