# coding:utf-8
import os 
import sys 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

import torch
print('at Tester')
print("gpu:",torch.cuda.get_device_name(torch.cuda.current_device()))

# if torch.cuda.is_available():
#     dev = "cuda"
# else:
#     dev = "cpu"
# device = torch.device(dev)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device=torch.device(device)
# cuda1 = torch.device('cuda:0')
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

# print(torch.cuda.get_device_name(1))

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
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import sys 

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testRel.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]


        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkLeftMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkRightMRR.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float
        self.lib.getTestLinkLeftMRR.restype = ctypes.c_float
        self.lib.getTestLinkRightMRR.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def test_one_step_by_index(self, data,index):        
        return self.model.predict_by_index({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        },index)

    def run_link_prediction(self,relation=None,type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        count=0
        for index, [data_head, data_tail] in enumerate(training_range):
            r = int(data_head['batch_r'][0])
            if relation == None: 
                count+=1
                score = self.test_one_step(data_head)
                self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
                score = self.test_one_step(data_tail)
                self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
            elif relation!=None and relation == r: 
                count+=1
                score = self.test_one_step(data_head)
                self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
                score = self.test_one_step(data_tail)
                self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain, count)
        sys.stdout.flush()

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        left_mrr = self.lib.getTestLinkLeftMRR(type_constrain)
        right_mrr = self.lib.getTestLinkRightMRR(type_constrain)
        return mrr, mr, hit10, hit3, hit1,left_mrr,right_mrr

    # def run_link_prediction(self,relation=None,type_constrain = False):
    #     self.lib.initTest()
    #     self.data_loader.set_sampling_mode('link')
    #     if type_constrain:
    #         type_constrain = 1
    #     else:
    #         type_constrain = 0
    #     training_range = tqdm(self.data_loader)
    #     count=0
    #     file=open("top10_ConvKB_FB15K237.txt",'w')
    #     for index, [data_head, data_tail] in enumerate(training_range):
    #         h = data_tail['batch_h']
    #         t = data_head['batch_t']
    #         r = data_head['batch_r']
    #         file.write("positive: %d %d %d \n"%(h,r,t))
    #         r = int(data_head['batch_r'][0])
    #         if relation == None: 
    #             count+=1
    #             score = self.test_one_step(data_head)
    #             negatives=extract_negatives(score,h,t,r,'corrupt_head')
    #             file.write("Negative Heads\n")
    #             for i in negatives:
    #                 file.write("%d %d %d \n"%(i,r,t))
    #             self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
    #             score = self.test_one_step(data_tail)
    #             negatives=extract_negatives(score,h,t,r,'corrupt_tail')
    #             file.write("Negative Tails\n")
    #             for i in negatives:
    #                 file.write("%d %d %d \n"%(h,r,i))
    #             self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
    #         elif relation!=None and relation == r: 
    #             count+=1
    #             score = self.test_one_step(data_head)
    #             self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
    #             score = self.test_one_step(data_tail)
    #             self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
    #     self.lib.test_link_prediction(type_constrain, count)
    #     sys.stdout.flush()

    #     mrr = self.lib.getTestLinkMRR(type_constrain)
    #     mr = self.lib.getTestLinkMR(type_constrain)
    #     hit10 = self.lib.getTestLinkHit10(type_constrain)
    #     hit3 = self.lib.getTestLinkHit3(type_constrain)
    #     hit1 = self.lib.getTestLinkHit1(type_constrain)
    #     left_mrr = self.lib.getTestLinkLeftMRR(type_constrain)
    #     right_mrr = self.lib.getTestLinkRightMRR(type_constrain)
    #     return mrr, mr, hit10, hit3, hit1,left_mrr,right_mrr

    def run_link_prediction_by_index(self,type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
   
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step_by_index(data_head,(index*2))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step_by_index(data_tail,((index*2)+1))
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)
        sys.stdout.flush()

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        return mrr, mr, hit10, hit3, hit1

    def run_link_prediction_for_ranks(self,type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)

        ranks=np.zeros((1086,75492))
        e=torch.tensor([i for i in range(75492)])
        test_triples=np.zeros((1086,3))
   
   
        for index, [data_head, data_tail] in enumerate(training_range):
            print(index)
            sys.stdout.flush()
            h = data_tail['batch_h'][0]
            t = data_head['batch_t'][0]
            r = data_head['batch_r'][0]
           
            test_triples[index*2]=np.array(([h,r,t]))
            test_triples[(index*2)+1]=np.array(([t,r,h]))
            # predicting tails
            score = self.test_one_step(data_tail)
            score1=torch.tensor(score)
            score1_ranks=torch.argsort(score1,descending=False)
            ranks1=torch.where(score1_ranks==e[:,None])[1]+1
            ranks1=ranks1.to(torch.float)
            ranks1=ranks1.cpu().data.numpy()
            self.lib.testTail(ranks1.__array_interface__["data"][0], index, type_constrain)
            ranks[index*2]=ranks1

            # predicting heads
            score = self.test_one_step(data_head)
            score1=torch.tensor(score)
            score1_ranks=torch.argsort(score1,descending=False)
            ranks1=torch.where(score1_ranks==e[:,None])[1]+1
            ranks1=ranks1.to(torch.float)
            ranks1=ranks1.cpu().data.numpy()
            self.lib.testHead(ranks1.__array_interface__["data"][0], index, type_constrain)
            ranks[(index*2)+1]=ranks1
         

        np.save("NELL-995_valid_triples.npy",test_triples)
        np.save("NELL-995_RotatE_valid_ranks.npy",ranks)
        
        self.lib.test_link_prediction(type_constrain)
        sys.stdout.flush()

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        return mrr, mr, hit10, hit3, hit1

    def run_link_prediction_for_scores(self,type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)

        scores=np.zeros((40932,14541))

        for index, [data_head, data_tail] in enumerate(training_range):
            sys.stdout.flush()
            h = data_tail['batch_h'][0]
            t = data_head['batch_t'][0]
            r = data_head['batch_r'][0]
           
            # predicting tails
            score = self.test_one_step(data_head)
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            scores[index*2]=-score

            # predicting heads
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
            scores[(index*2)+1]=-score
         
        np.save("Test_Scores/FB15K237_ConvKB_test_scores.npy",scores)
        
        self.lib.test_link_prediction(type_constrain)
        sys.stdout.flush()

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        return mrr, mr, hit10, hit3, hit1

    #my version
    # def run_link_prediction(self, type_constrain = False):
    #     file=open("top10_convkb_FB15K237.txt",'w')

    #     self.lib.initTest()
    #     self.data_loader.set_sampling_mode('link')
    #     if type_constrain:
    #         type_constrain = 1
    #     else:
    #         type_constrain = 0
    #     training_range = tqdm(self.data_loader)
    #     for index, [data_head, data_tail] in enumerate(training_range):
    #         h = data_tail['batch_h']
    #         t = data_head['batch_t']
    #         r = data_head['batch_r']
    #         file.write("positive: %d %d %d \n"%(h,t,r))
    #         score = self.test_one_step(data_head)
    #         negatives=extract_negatives(score,h,t,r,'corrupt_head')
    #         file.write("Negative Heads\n")
    #         for i in negatives:
    #             file.write("%d %d %d \n"%(i,t,r))
    #         self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
    #         score = self.test_one_step(data_tail)
    #         negatives=extract_negatives(score,h,t,r,'corrupt_tail')
    #         self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
    #         file.write("Negative Tails\n")
    #         for i in negatives:
    #             file.write("%d %d %d \n"%(h,i,r))
    #     file.close()
    #     self.lib.test_link_prediction(type_constrain)
    #     sys.stdout.flush()

    #     mrr = self.lib.getTestLinkMRR(type_constrain)
    #     mr = self.lib.getTestLinkMR(type_constrain)
    #     hit10 = self.lib.getTestLinkHit10(type_constrain)
    #     hit3 = self.lib.getTestLinkHit3(type_constrain)
    #     hit1 = self.lib.getTestLinkHit1(type_constrain)
    #     return mrr, mr, hit10, hit3, hit1

    def run_relation_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('relation')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_rel] in enumerate(training_range):
            score = self.test_one_step(data_rel)
            self.lib.testRel(score.__array_interface__["data"][0], index)
            
        self.lib.test_relation_prediction(type_constrain)
        sys.stdout.flush()

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        print('here...')
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')

        score = []
        ans = []
        training_range = tqdm(self.data_loader)

        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod

def extract_negatives(score,h,t,r,tag):
    score=score.flatten()
    sorted_ind=score.argsort()
    if tag=='corrupt_head':
        key=h 
    else:
        key=t
    # rank=list(sorted_ind).index(key)
    # print ("RANK=",rank)
    negatives = sorted_ind[:10]
    return negatives

