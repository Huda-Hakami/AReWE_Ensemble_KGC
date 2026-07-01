# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss

class SigmoidLoss(Loss):

	def __init__(self, adv_temperature = None):
		super(SigmoidLoss, self).__init__()
		self.criterion = nn.LogSigmoid()
		self.criterion2=nn.BCEWithLogitsLoss()
		self.sig=nn.Sigmoid()
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False

	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
		else:
			# print("Pos",p_score.mean(),p_score.std())
			# print("Neg",n_score.mean(),p_score.std())

			# print(self.sig(n_score).mean())

			return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2
			# max_n_score=torch.amax(n_score, 1)
			# p_score=p_score.flatten()
			# return -(torch.log(p_score).mean()+torch.log(-n_score).mean()) /2
			
			# return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2
			
			# pos_loss=self.criterion(p_score).mean()
			# neg_loss=self.criterion(-n_score).mean()
			# # print("POS loss",-pos_loss)
			# # print("NEG loss",-neg_loss)
			# return -( pos_loss+ neg_loss )/2

			# z=torch.cat((p_score,n_score),dim=1)
			# prob=F.softmax(z,dim=1)[:,0]
			# return -(torch.log(prob).mean())

			# pos_exp = torch.exp(p_score).view(-1)
			# neg_exp = torch.exp (n_score)
			# softmax_score = pos_exp / (torch.sum(neg_exp,dim=1) + pos_exp)
			# loss = -(torch.log(softmax_score).mean())
			# return loss
			
	# def forward (self,logits,labels):
	# 	labels=labels.to(device)
	# 	return self.criterion2(logits,labels)

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()