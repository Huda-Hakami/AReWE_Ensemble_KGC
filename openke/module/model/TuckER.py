import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import numpy as np

    
class TuckER(Model):

    def __init__(self,ent_tot, rel_tot, dim = 100, kernel_size=1, out_channels=64,drop_prob=0.5):
        super(TuckER, self).__init__(ent_tot, rel_tot)
        
        self.kernel_size=kernel_size
        self.out_channels=out_channels
        self.drop_prob=drop_prob
        self.dim = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim) 
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            


        self.W = torch.nn.Parameter(torch.tensor(
            np.random.uniform(-1, 1, (self.dim, self.dim, self.dim)), 
            dtype=torch.float, device="cuda", requires_grad=True))

        self.hidden_dropout1 = torch.nn.Dropout(drop_prob)
        self.hidden_dropout2 = torch.nn.Dropout(drop_prob)
        self.bn0 = torch.nn.BatchNorm1d(self.dim)
        self.bn1 = torch.nn.BatchNorm1d(self.dim)

    def _calc(self, h, r, t):

        W_mat  = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat  = W_mat.view(-1, h.size(1), h.size(1)) 
        W_mat  = self.hidden_dropout1(W_mat)
        hr_vec = torch.bmm(h.unsqueeze(1), W_mat) 
        hr_vec = self.bn1(hr_vec.squeeze(1)).unsqueeze(1)
        hr_vec = self.hidden_dropout2(hr_vec)

        score = torch.bmm(hr_vec, t.view(-1,self.dim,1)).squeeze(1)             # [B, N]

        score = torch.sigmoid(score)
        
        return score.flatten()



    def forward(self,data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
  
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)

        score = self._calc(h, r, t)

        return score


    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)

        return l2_reg

    def predict(self,data):
        score = -self.forward(data)
        return score.cpu().data.numpy()
