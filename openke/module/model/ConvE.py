import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class ConvE(Model):
	def __init__(self, ent_tot, rel_tot,dim = 200,input_drop=0.2, hid_drop=0.3,feat_drop=0.2,embed_shape1=20,use_bias=True,hidden_size=9728):
        super(ConvE, self).__init__(ent_tot, rel_tot)
        self.dim = dim
        self.input_drop=input_drop
        self.hid_drop=hidden_drop
        self.feat_drop=feat_drop
        self.embed_shape1=embed_shape1
        self.use_bias=use_bias
        self.hidden_size=hidden_size

        self.emb_e = nn.Embedding(self.ent_tot, self.dim, padding_idx=0)
        self.emb_rel = nn.Embedding(self.rel_tot, self.dim, padding_idx=0)
        self.inp_drop = nn.Dropout(self.input_drop)
        self.hidden_drop = nn.Dropout(self.hid_drop)
        self.feature_map_drop = nn.Dropout2d(self.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = self.embed_shape1
        self.emb_dim2 = self.dim // self.emb_dim1

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=self.use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = BatchNorm1d(self.dim)
        self.register_parameter('b', Parameter(torch.zeros(self.ent_tot)))
        self.fc = nn.Linear(self.hidden_size,self.dim)
        print(self.ent_tot, self.rel_tot)

        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        
    def forward(self, data):
    	batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
        h= self.emb_e(batch_h).view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = self.emb_rel(batch_r).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([h, r], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()
