
# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

from .Strategy import Strategy
import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
class NegativeSamplingAReWE(Strategy):

	def __init__(self, model = None, loss = None, weighting_strategy=None, regul_rate1 = 0.0, regul_rate2 = 0.0, l3_regul_rate = 0.0):
		super(NegativeSamplingAReWE, self).__init__()
		self.model = model
		self.loss = loss
		self.weighting_strategy = weighting_strategy
		self.batch_size = self.model.batch_size
		self.regul_rate1 = regul_rate1
		self.regul_rate2 = regul_rate2
		self.l3_regul_rate = l3_regul_rate
		self.neg_examples = self.model.neg_examples
		self.logSoftmax=nn.LogSoftmax(dim=1)

	
	def _get_positive_score(self, score, bsize):
		positive_score = score[:bsize]
		positive_score = positive_score.view(-1, bsize).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score, bsize):
		negative_score = score[bsize:]
		negative_score = negative_score.view(-1, bsize).permute(1, 0)
		return negative_score
	

	def forward(self, data, bsize, mrr = None):

		score1,score2,score3,score4,score5 = self.model(data, bsize)

		score1ls = self.min_max_normalize_local(score1,bsize)
		score1ls =  1 - score1ls #Transfer distance to similarity
		score2ls = self.min_max_normalize_local(score2,bsize)
		score3ls = self.min_max_normalize_local(score3,bsize)
		score4ls = self.min_max_normalize_local(score4,bsize)
		score5ls = self.min_max_normalize_local(score5,bsize)

		avg_score = (score1ls+score2ls+score3ls+score4ls+score5ls)/5.0

		score=torch.cat((score1ls,score2ls,score3ls,score4ls,score5ls),0).view(-1,bsize,self.neg_examples+1) #shape num_of_models, bsize, pos+neg_examples
	
		batch_r = data["batch_r"][:bsize]

		dic = {}

		model_scores = [
			score1ls,
			score2ls,
			score3ls,
			score4ls,
			score5ls,
			avg_score,
		]

		for sample_idx, relation in enumerate(batch_r):
			relation_id = relation.item()

			dic.setdefault(
				relation_id,
				{model_idx: [] for model_idx in range(len(model_scores))}
			)

			for model_idx, scores in enumerate(model_scores):
				dic[relation_id][model_idx].append(
					scores[sample_idx].cpu().data.numpy().tolist()
				)


		for rel in dic:
			for i in dic[rel]:
				dic[rel][i] = torch.FloatTensor(dic[rel][i])

		current_estimated_weights=torch.zeros([self.model.rel_tot,5],dtype=torch.float).cuda()


		for relation_id, predictions in dic.items():

			model_weights = []

			for model_idx in range(5):
				weight = self.compute_relation_weight(
					predictions[model_idx]
				)
				model_weights.append(weight)

			model_weights = torch.stack(model_weights)
			model_weights = F.softmax(model_weights, dim=0)

			current_estimated_weights[relation_id] = model_weights

		weights_mat=torch.zeros([bsize,5],dtype=torch.float).cuda()
		for k,rel in enumerate(batch_r):
			# weights_mat[k]=weights_dic[rel.item()]
			weights_mat[k] = current_estimated_weights[rel]

			
		loss_res = self.loss(avg_score,score1ls,score2ls,score3ls,score4ls,score5ls,weights_mat)

		
		if self.regul_rate1 != 0:
			loss_res += self.regul_rate1 * self.model.regularization(data,bsize)
		if self.regul_rate2 != 0:
			loss_res += self.regul_rate2 * self.model.regularization2(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res


	def min_max_normalize_local(self,score,bsize):
		p_score = self._get_positive_score(score, bsize)
		n_score = self._get_negative_score(score, bsize)
		score=torch.cat((p_score,n_score),1)
		min_=torch.min(score,dim=1).values
		min_=min_.view(-1,1)
		max_=torch.max(score,dim=1).values
		max_=max_.view(-1,1)
		score=(score-min_)/(max_-min_)
		return score

	def min_max_normalize_global(self,score,bsize):
		p_score = self._get_positive_score(score, bsize)
		n_score = self._get_negative_score(score, bsize)
		score=torch.cat((p_score,n_score),1)
		min_=torch.min(score)
		max_=torch.max(score)
		score=(score-min_)/(max_-min_)
		return score

	def min_max_normalize_RelSpecific(self,score,data,bsize):

		p_score = self._get_positive_score(score, bsize)
		n_score = self._get_negative_score(score, bsize)
		score=torch.cat((p_score,n_score),1)

		batch_r = data['batch_r'][:bsize]
		dic={}
		for k,rel in enumerate(batch_r):
			dic.setdefault(rel.item(),[])
			s = score[k].cpu().data.numpy()
			dic[rel.item()].append(list(s))

		for rel in dic:
			dic[rel] = torch.FloatTensor(dic[rel])

		min_max={rel:{'min':0.0,'max':0.0} for rel in dic}

		for rel in dic: 
			min_=torch.min(dic[rel])
			max_=torch.max(dic[rel])
			min_max[rel]['min']=min_ 
			min_max[rel]['max']=max_ 

		for k,rel in enumerate(batch_r):
			rel=rel.item()
			score[k]=(score[k]-min_max[rel]['min'])/(min_max[rel]['max']-min_max[rel]['min'])

		return score

	def standard_normalize(self,score,bsize):
		p_score = self._get_positive_score(score, bsize)
		n_score = self._get_negative_score(score, bsize)
		score=torch.cat((p_score,n_score),1)
		m=torch.mean(score,dim=1)
		m=m.view(-1,1)
		std=torch.std(score,dim=1)
		std=std.view(-1,1)
		score=(score-m)/(std)
		return score

	def compute_relation_weight(self,predictions):
		if self.weighting_strategy == "score_based":
			return predictions[:, 0].mean()

		elif strategy == "mrr_based":
			ranks = (
				torch.argsort(
					torch.argsort(predictions, dim=1, descending=True),
					dim=1,
					descending=False,
				)
				+ 1
			)
			return (1.0 / ranks[:, 0]).mean()

		else:
			raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

	def get_scores(self,score,bsize):
		p_score = self._get_positive_score(score, bsize)
		n_score = self._get_negative_score(score, bsize)
		score=torch.cat((p_score,n_score),1)
		return score


	
