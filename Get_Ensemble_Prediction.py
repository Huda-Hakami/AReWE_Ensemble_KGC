import openke
from openke.config import Trainer, Tester
from openke.module.model import EnsemblePrediction
from openke.data import TrainDataLoader,TestDataLoader
import sys 
import numpy as np 

print ("MetaEmbed Model")
data_folder="NELL-995"
sys.stdout.flush()

train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/%s/"%data_folder, 
	nbatches = 2000,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
)
# define the model
embeddings = EnsemblePrediction(
	data_folder = data_folder,
	batch_size = train_dataloader.get_batch_size(),
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
)

embeddings.save_checkpoint('./checkpoint/embeddings_%s.ckpt'%data_folder)
# test the model
embeddings.load_checkpoint('./checkpoint/embeddings_%s.ckpt'%data_folder)
print ("==========================================")
print ("TEST_All") 
sys.stdout.flush()

# dataloader for test
if data_folder=='FB15K237' or data_folder=='WN18RR':
	test_folder='test2id.txt'
elif data_folder=='NELL-995':
	test_folder='test2id_refined.txt'

test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,"n-n.txt", "link")
tester1 = Tester(model = embeddings, data_loader = test_dataloader_all, use_gpu = True)
tester1.run_link_prediction(type_constrain = True)
# tester1.run_link_prediction(type_constrain = True)
# acc,thre=tester1.run_triple_classification()
# print (acc,thre)
sys.stdout.flush()


# test the model on TAIL RELATIONS
# print ("==========================================")
# print ("TEST_All") 
# sys.stdout.flush()
# # dataloader for test

# if data_folder=='FB15K237' or data_folder=='WN18RR':
# 	test_folder='test2id.txt'
# elif data_folder=='NELL-995':
# 	test_folder='test2id_refined.txt'

# rel_dic={}
# with open("./benchmarks/%s/valid2id.txt"%data_folder) as valid: 
# 	for line in valid: 
# 		line=line.strip().split()
# 		if len(line)!=1:
# 			h,t,r = int(line[0]),int(line[1]),int(line[2])
# 			rel_dic.setdefault(r,set())
# 			rel_dic[r].add((h,t))
# list_=[]
# for rel in rel_dic:
# 	list_.append((len(rel_dic[rel]),rel))
# list_.sort(reverse=True)
# ordered_rel=[x[1] for x in list_]
# ind=int((len(ordered_rel)*50)/100)
# tail_relations = ordered_rel[-ind:]
# head_relaitons = ordered_rel[:50]
# body_relations=ordered_rel[50:110]
# print(len(tail_relations)+len(head_relaitons)+len(body_relations))

# test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,test_folder, "link")
# tester1 = Tester(model = embeddings, data_loader = test_dataloader_all, use_gpu = True)
# MRR=[]
# MR=[]
# Hits1=[]
# Hits3=[]
# Hits10=[]
# for rel in ordered_rel:
# 	print("Relation",rel)
# 	mrr, mr, hit10, hit3, hit1,left_mrr,right_mrr=tester1.run_link_prediction(relation=rel,type_constrain = False)
# 	if not np.isnan(mrr):
# 		MRR.append(mrr)
# 		MR.append(mr)
# 		Hits1.append(hit1)
# 		Hits3.append(hit3)
# 		Hits10.append(hit10)
# print("Macro-average Results..")
# print("MRR=%f, MR=%f, Hits@10=%f, Hists@3=%f, Hits@1=%f"%(np.mean(MRR),np.mean(MR),np.mean(Hits10),np.mean(Hits3),np.mean(Hits1)))
# sys.stdout.flush()



	

