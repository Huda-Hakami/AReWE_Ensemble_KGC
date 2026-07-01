import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import numpy as np

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
    
class ConvKB(Model):

    def __init__(self, data_folder,ent_tot, rel_tot, dim = 100, kernel_size=1, out_channels=64,drop_prob=0.5, use_init_embeddings=False):
        super(ConvKB, self).__init__(ent_tot, rel_tot)
        
        self.kernel_size=kernel_size
        self.out_channels=out_channels
        self.use_init_embeddings=use_init_embeddings
        self.drop_prob=drop_prob
        self.dim = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim) 
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        
        if self.use_init_embeddings==False:  # random initialization
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            
        else:  # initialize with TransE
            lstent = []
            lstrel = []
            with open("/users/hahakami/meta-kge/OpenKE-OpenKE-PyTorch/benchmarks/%s/relation2vec%d.init"%(data_folder,self.dim)) as f:
                for line in f:
                    tmp = [float(val) for val in line.strip().split()]
                    lstrel.append(tmp)
            with open("/users/hahakami/meta-kge/OpenKE-OpenKE-PyTorch/benchmarks/%s/entity2vec%d.init"%(data_folder,self.dim)) as f:
                for line in f:
                    tmp = [float(val) for val in line.strip().split()]
                    lstent.append(tmp)
          
            init_ent_embs=np.array(lstent, dtype=np.float32)
            init_rel_embs=np.array(lstrel, dtype=np.float32)
            self.ent_embeddings.weight.data = torch.from_numpy(init_ent_embs)
            self.rel_embeddings.weight.data = torch.from_numpy(init_rel_embs)

            # TransE_embeddings=np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreTrained_KGEs/transe_embeddings_%s.npy"%data_folder,allow_pickle=True,encoding = 'bytes').item()
            # transe_ent_weight=torch.FloatTensor(TransE_embeddings[b'ent_embeddings.weight'])
            # transe_rel_weight=torch.FloatTensor(TransE_embeddings[b'rel_embeddings.weight'])
            # self.ent_embeddings=nn.Embedding.from_pretrained(transe_ent_weight)
            # self.rel_embeddings=nn.Embedding.from_pretrained(transe_rel_weight)
            # self.dim=transe_ent_weight.shape[1]

        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, self.out_channels, (self.kernel_size, 3))  # kernel size x 3
        self.conv2_bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(self.drop_prob)
        self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.fc_layer = nn.Linear((self.dim - self.kernel_size + 1) * self.out_channels, 1, bias=False)

        self.criterion = nn.Softplus()

        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)

    def _calc(self, h, r, t):

        h = h.unsqueeze(1) # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.dim - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)

        return score


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
    def forward2(self,data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        

        if mode == 'head_batch':
            r=torch.cat([r]*h.shape[0])
            t=torch.cat([t]*h.shape[0])
        elif mode=='tail_batch':
            h=torch.cat([h]*t.shape[0])
            r=torch.cat([r]*t.shape[0])
        elif mode=='rel_batch':
            h=torch.cat([h]*r.shape[0])
            t=torch.cat([t]*r.shape[0])

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
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        return l2_reg

    def predict(self,data):
        score = -self.forward2(data)
        return score.cpu().data.numpy()
