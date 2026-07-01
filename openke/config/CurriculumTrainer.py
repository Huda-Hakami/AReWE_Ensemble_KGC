
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print('at CurriculumTrainer')
print('gpu:', torch.cuda.get_device_name(torch.cuda.current_device()))
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
from openke.data import TestDataLoader
from openke.config.Tester import Tester
import numpy as np
class CurriculumTrainer(object):

    def __init__(self, model=None, data_loader=None, train_times=1000, alpha=0.5, use_gpu=True, opt_method='sgd', data_folder=None, negEnt=None, rel_batch = False, save_steps=None, early_stop=None, checkpoint_dir=None):
        self.work_threads = 8
        self.train_times = train_times
        self.opt_method = opt_method
        self.data_folder = data_folder
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        self.model = model
        self.negEnt=negEnt
        self.rel_batch = rel_batch
        print('self.model=', self.model)
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.early_stop = early_stop
        self.checkpoint_dir = checkpoint_dir

    def train_one_step(self, data, epoch,rel_batch_size, mrr=None):
        self.optimizer.zero_grad()
        loss = self.model({'batch_h':self.to_var(data['batch_h'], self.use_gpu), 
         'batch_t':self.to_var(data['batch_t'], self.use_gpu), 
         'batch_r':self.to_var(data['batch_r'], self.use_gpu), 
         'batch_y':self.to_var(data['batch_y'], self.use_gpu), 
         'mode':data['mode']},rel_batch_size, mrr)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) #added for convkb

        self.optimizer.step()

        # with torch.no_grad():
        #     w = self.model.model.weights
        #     sum_=torch.sum(w,dim=1).view(-1,1)
        #     w=w/sum_
        #     self.model.model.weights.data = w
       

        # w = self.model.model.weights.weight.data
        # w = w.clamp(0,1)
        # sum_=torch.sum(w,dim=1).view(-1,1)
        # w=w/sum_
        # self.model.model.weights.weight.data = w

        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        # for name,param in self.model.named_parameters():
        #     print (name,param.requires_grad)
        # exit()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")
        training_range = tqdm(range(self.train_times))
        patience=1
        count=0
        best_performance=1e-7
        relations = self.data_loader.get_rel_tot()
        
        data_by_rel={}
        for data in self.data_loader:
            data['batch_h'] = data['batch_h'].flatten()
            data['batch_t'] = data['batch_t'].flatten()
            data['batch_r'] = data['batch_r'].flatten()
            data['batch_y'] = data['batch_y'].flatten()

            if data['mode']=='tail_batch':
                data['batch_h']=np.concatenate([data['batch_h']]*(self.negEnt+1))
                data['batch_r']=np.concatenate([data['batch_r']]*(self.negEnt+1))
            elif data['mode']=='head_batch':
                data['batch_t']=np.concatenate([data['batch_t']]*(self.negEnt+1))
                data['batch_r']=np.concatenate([data['batch_r']]*(self.negEnt+1))
            # organize data by relation types
            for r in range(len(data['batch_r'])):
                rel_id= data['batch_r'][r]
                data_by_rel.setdefault(rel_id,{'batch_h':[],'batch_t':[],'batch_r':[],'batch_y':[],'mode':''})
                data_by_rel[rel_id]['batch_h'].append(data['batch_h'][r])
                data_by_rel[rel_id]['batch_t'].append(data['batch_t'][r])
                data_by_rel[rel_id]['batch_r'].append(data['batch_r'][r])
                data_by_rel[rel_id]['batch_y'].append(data['batch_y'][r])
                data_by_rel[rel_id]['mode']=data['mode']

        print (data_by_rel[0]['batch_h'])
        print (data_by_rel[0]['batch_t'])
        print (data_by_rel[0]['batch_r'])
        print (data_by_rel[0]['batch_y'])
        print (data_by_rel[0]['mode'])
        print(len(data_by_rel[0]['batch_h']))
        exit()
        mrr = np.load("/LOCAL2/huda/meta-kge/OpenKE-OpenKE-PyTorch/PreWeights/PreRelWeightsPerKGE_%s_mrr_valid.npy"%self.data_folder)
        std_=mrr.std(axis=1)
        top_var=(std_).argsort()
        jump = 10
        initial_dic={}
        sub=match.ceil((len(top_var)*jump)/100)
        conisered_rel = top_var[:sub]
        
        for epoch in training_range: 
            if epoch <=jump:
               continue 

            else: 
                jump+=10

        for epoch in training_range:

            res = 0.0
            t_init = time.time()

            if self.save_steps and epoch != 0 and (epoch%self.save_steps) == 0:
                print("measure performance of relation types appearing in the mini-batch")
                MRR, MRR_l, MRR_r = self.measure_performance(epoch-1)
                w = {'mrr':MRR,'left':MRR_l,'right':MRR_r}
                print("at epoch%d, weights has been changed"%epoch)

            for data in self.data_loader:
                data['batch_h'] = data['batch_h'].flatten()
                data['batch_t'] = data['batch_t'].flatten()
                data['batch_r'] = data['batch_r'].flatten()
                data['batch_y'] = data['batch_y'].flatten()

                if data['mode']=='tail_batch':
                    data['batch_h']=np.concatenate([data['batch_h']]*(self.negEnt+1))
                    data['batch_r']=np.concatenate([data['batch_r']]*(self.negEnt+1))
                elif data['mode']=='head_batch':
                    data['batch_t']=np.concatenate([data['batch_t']]*(self.negEnt+1))
                    data['batch_r']=np.concatenate([data['batch_r']]*(self.negEnt+1))

                rel_batch_size=int(len(data['batch_r'])/(1+self.negEnt))


                if not self.rel_batch: 
                    if self.save_steps: 
                        if epoch >=0 and epoch<self.save_steps :
                            mrr  = {r:1/relations for r in range(relations)}
                        elif data['mode'] =='head_batch':
                            mrr = w ['left']
                        elif data['mode'] == 'tail_batch':
                            mrr = w ['right']
                        elif data['mode'] == 'normal':
                            mrr = w ['mrr']
                        loss = self.train_one_step(data, epoch,rel_batch_size, mrr)
                    else: 
                        loss = self.train_one_step(data, epoch,rel_batch_size)
                    res += loss
                else:
                    data2={}
                    # print("rel batch size",rel_batch_size)
                    for r in range(len(data['batch_r'])):

                        data2.setdefault(data['batch_r'][r],{'batch_h':[],'batch_t':[],'batch_r':[],'batch_y':[],'mode':''})
                        data2[data['batch_r'][r]]['batch_h'].append(data['batch_h'][r])
                        data2[data['batch_r'][r]]['batch_t'].append(data['batch_t'][r])
                        data2[data['batch_r'][r]]['batch_r'].append(data['batch_r'][r])
                        data2[data['batch_r'][r]]['batch_y'].append(data['batch_y'][r])
                        data2[data['batch_r'][r]]['mode']=data['mode']
                    
                    for r in data2:

                        data2[r]['batch_h']=np.array(data2[r]['batch_h']).flatten()
                        data2[r]['batch_t']=np.array(data2[r]['batch_t']).flatten()
                        data2[r]['batch_r']=np.array(data2[r]['batch_r']).flatten()
                        data2[r]['batch_y']=np.array(data2[r]['batch_y']).flatten()
                
                    for rel in data2: 
                        rel_batch_size=int(len(data2[rel]['batch_r'])/(1+self.negEnt))
                        # print(rel,"rel batch size",rel_batch_size)
                        # loss = self.train_one_step(data2[rel], epoch,rel_batch_size)
                        # res += loss

                        mini_batch_size= 1e7
                        if rel_batch_size > mini_batch_size:
                            # print(rel,"rel batch size",rel_batch_size)
                            dic={'batch_h':[],'batch_t':[],'batch_r':[],'batch_y':[],'mode':''}
                            dic['mode']=data2[rel]['mode']
                            p_h = data2[rel]['batch_h'][:rel_batch_size]
                            p_t = data2[rel]['batch_t'][:rel_batch_size]
                            p_r = data2[rel]['batch_r'][:rel_batch_size]
                            p_y = data2[rel]['batch_y'][:rel_batch_size]

                            n_h=data2[rel]['batch_h'][rel_batch_size:]
                            n_h=[n_h[x:x+len(p_h)] for x in range(0, len(n_h), len(p_h))]

                            n_t=data2[rel]['batch_t'][rel_batch_size:]
                            n_t=[n_t[x:x+len(p_t)] for x in range(0, len(n_t), len(p_t))]

                            n_r=data2[rel]['batch_r'][rel_batch_size:]
                            n_r=[n_r[x:x+len(p_r)] for x in range(0, len(n_r), len(p_r))]

                            n_y=data2[rel]['batch_y'][rel_batch_size:]
                            n_y=[n_y[x:x+len(p_y)] for x in range(0, len(n_y), len(p_y))]


                            for minibatch in self.next_batch(mini_batch_size,p_h,p_t,p_r,p_y,n_h,n_t,n_r,n_y):

                                size=len(minibatch[0])
                                # print(size)

                                dic['batch_h'],dic['batch_t'],dic['batch_r'],dic['batch_y']=minibatch[0],minibatch[1],minibatch[2],minibatch[3]
                                for k in minibatch[4]:
                                    dic['batch_h']=np.append(dic['batch_h'],np.array(k))
                                for k in minibatch[5]:
                                    dic['batch_t']=np.append(dic['batch_t'],np.array(k))
                                for k in minibatch[6]:
                                    dic['batch_r']=np.append(dic['batch_r'],np.array(k))
                                for k in minibatch[7]:
                                    dic['batch_y']=np.append(dic['batch_y'],np.array(k))

                                loss = self.train_one_step(dic, epoch,size)
                                res += loss                    
                        else: 
                            loss = self.train_one_step(data2[rel], epoch,rel_batch_size)
                            res += loss
                
            t_end = time.time()
            training_range.set_description('Epoch %d | loss: %f | time: %f' % (epoch, res, t_end - t_init))
            print('Epoch %d | loss: %f' % (epoch, res))
            sys.stdout.flush()
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print('Epoch %d has finished, saving...' % epoch)
                self.model.model.save_checkpoint(self.checkpoint_dir + '-' + str(self.data_folder) + '-' + str(epoch) + '.ckpt')

            # ...early stopping
            if self.early_stop and (epoch + 1) % 5 == 0:
                if self.data_folder=='NELL-995':
                    test_folder="test2id_refined.txt"
                else: 
                    test_folder="test2id.txt"
                self.model.model.save_checkpoint(self.checkpoint_dir + '-' + str(self.data_folder) + '-' + str(epoch) + '.ckpt')
                self.model.model.load_checkpoint(self.checkpoint_dir + '-' + str(self.data_folder) + '-' + str(epoch) + '.ckpt')
                test_dataloader = TestDataLoader('./benchmarks/%s/' % self.data_folder, test_folder, 'link')
                tester = Tester(model=self.model.model, data_loader=test_dataloader, use_gpu=True)
                mrr, mr, hit10, hit3, hit1, left_mrr, right_mrr=tester.run_link_prediction()
                print("Valid mrr=",mrr)
                sys.stdout.flush()
                if mrr>best_performance:
                    best_performance=mrr
                    self.model.model.save_checkpoint('./checkpoint/metaembed_%s.ckpt'%self.data_folder)
                    print ("The model parameters have been saved correctly...")
                    count=0
                else:
                    count+=1
                if count>=patience and (epoch+1)>50:
                    print("early stopping...")
                    break
                
    def measure_performance (self,epoch):
        if self.data_folder=="NELL-995":
            relations=[96, 97, 99, 164, 35, 103, 91, 155, 25, 122, 27, 126]
            mrr = {r:0.0 for r in relations}
            left_mrr = {r:0.0 for r in relations}
            right_mrr = {r:0.0 for r in relations}
        else:
            mrr = {r:0.0 for r in range(self.data_loader.get_rel_tot())}
            left_mrr = {r:0.0 for r in range(self.data_loader.get_rel_tot())}
            right_mrr = {r:0.0 for r in range(self.data_loader.get_rel_tot())}
        self.model.model.load_checkpoint(self.checkpoint_dir + '-' + str(self.data_folder) + '-' + str(epoch) + '.ckpt')
        test_dataloader = TestDataLoader('./benchmarks/%s/' % self.data_folder, 'valid2id.txt', 'link')
        tester = Tester(model=self.model.model, data_loader=test_dataloader, use_gpu=True)
        for rel in mrr: 
            MRR, MR, hit10, hit3, hit1, L_MRR, R_MRR = tester.run_link_prediction(relation=rel)
            print("rel:%d, MRR:%f"%(rel,MRR))
            print("rel:%d, LeftMRR:%f"%(rel,L_MRR))
            print("rel:%d, RightMRR:%f"%(rel,R_MRR))
            left_mrr[rel] = L_MRR
            right_mrr[rel] = R_MRR
            mrr[rel] = MRR
        return mrr, left_mrr, right_mrr

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_early_stop(self,early_stop):
        self.early_stop = early_stop

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def next_batch(self,bsize,p_h,p_t,p_r,p_y,n_h,n_t,n_r,n_y):
        # loop over our dataset in mini-batches of size `batchSize`
        for i in np.arange(0, len(p_h), bsize):
            # yield the current batched data 

            yield (p_h[i:i + bsize],p_t[i:i + bsize],p_r[i:i + bsize],p_y[i:i + bsize],
                    (n_h[j][i:i + bsize] for j in range(self.negEnt)),(n_t[j][i:i + bsize] for j in range(self.negEnt)),(n_r[j][i:i + bsize] for j in range(self.negEnt)),(n_y[j][i:i + bsize] for j in range(self.negEnt)))
# -------------------------------------------------------