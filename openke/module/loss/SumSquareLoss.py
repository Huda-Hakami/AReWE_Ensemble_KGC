import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss
# from torchmetrics import MeanSquaredLogError
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class SumSquareLoss(Loss):

	def __init__(self, adv_temperature = None, margin = 0.0):
		super(SumSquareLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score, weights):

		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			# mse=torch.nn.MSELoss(reduction='mean')
			# msle = MeanSquaredLogError()
			# msle.to(device)
			# return  mse(n_score,p_score)

			
			return ((p_score-n_score)**2).mean()
			# weights=weights.cuda()
			# return ((weights*(p_score-n_score))**2).mean()
			# return ((weights * (p_score - n_score) ** 2).sum() / weights.sum()).mean()
			# diff = (p_score - n_score) ** 2
			# return (diff * weights).sum() / weights.sum()

		# else:
		# 	return torch.max(-p_score+n_score, self.margin).sum(dim = -1).mean()
			
	
	def predict(self, p_score, n_score):

		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()