import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss

class MarginLoss(Loss):

	def __init__(self, adv_temperature = None, margin = 1.0):
		super(MarginLoss, self).__init__()

		self.criterion = nn.LogSigmoid()
		self.criterion2=nn.LogSoftmax(dim=1)
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

		self.margin_0 = nn.Parameter(torch.Tensor([0.0]))
		self.margin_0.requires_grad = False

		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, avg_score, score1,score2,score3,score4,score5,weights):
 

		p_score1=score1[:,0].view(-1,1)
		n_score1=score1[:,1:]

		p_score2=score2[:,0].view(-1,1)
		n_score2=score2[:,1:]

		p_score3=score3[:,0].view(-1,1)
		n_score3=score3[:,1:]

		p_score4=score4[:,0].view(-1,1)
		n_score4=score4[:,1:]

		p_score5=score5[:,0].view(-1,1)
		n_score5=score5[:,1:]

		loss1=F.relu(self.margin+torch.amax(n_score1,1)-p_score1.flatten()).view(-1,1)
		loss2=F.relu(self.margin+torch.amax(n_score2,1)-p_score2.flatten()).view(-1,1)
		loss3=F.relu(self.margin+torch.amax(n_score3,1)-p_score3.flatten()).view(-1,1)
		loss4=F.relu(self.margin+torch.amax(n_score4,1)-p_score4.flatten()).view(-1,1)
		loss5=F.relu(self.margin+torch.amax(n_score5,1)-p_score5.flatten()).view(-1,1)


		losses=torch.cat((loss1,loss2,loss3,loss4,loss5),1)
		relation_specific_loss=torch.mul(losses,weights).sum(1).mean()

		# ....avg_loss with top-1 neg
		p_avg = avg_score[:,0].view(-1,1)
		n_avg = avg_score[:,1:]
		max_n_avg=torch.amax(n_avg,1)
		p_avg=p_avg.flatten()
		avg_loss=F.relu(self.margin + max_n_avg - p_avg)
		avg_loss = avg_loss.mean()

		return avg_loss+ relation_specific_loss
    
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()
