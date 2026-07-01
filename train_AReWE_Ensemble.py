from openke.config import Trainer, Tester
from openke.module.model import OptimizedEnsemble
from openke.module.loss import  MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import PyTorchTrainDataLoader, TestDataLoader
import numpy as np 

data_folder="NELL-995"
negEnt=500
margin = 0.8
regul_rate1 = 0.0
negRel=0
sampling_mode ="cross"
Train=True  

train_dataloader = PyTorchTrainDataLoader(
	in_path = "./benchmarks/%s/"%data_folder, 
	tri_file = "valid2id.txt",
	batch_size= 64,
	threads = 1, 
	sampling_mode = sampling_mode, 
	bern_flag = 1,
	TC_NS = 0, 
	filter_flag = 1, 
	neg_ent = negEnt,
	neg_rel = negRel
)

print("batch_size",train_dataloader.get_batch_size())
print("number of batches",train_dataloader.nbatches)
print ("train_dataloader worked...")

# define the ensemble model
ensemble_model = OptimizedEnsemble(
	data_folder = data_folder,
	batch_size = train_dataloader.get_batch_size(),
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	neg_examples=negEnt+negRel,
	sampling_mode = sampling_mode
)

print ("the model is defined correctly...")

# define the loss function
model = NegativeSampling(
	model = ensemble_model, 
	loss = MarginLoss(margin = margin),
	weighting_strategy = 'score_based',  #or mrr_based
	regul_rate1 = regul_rate1, #l2 reg
	regul_rate2 = 0.0, #orthogonality 
)

early_stop=0
if Train:
	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 50, alpha = 0.001, use_gpu = True, opt_method = "adam", data_folder=data_folder, negEnt=negEnt, rel_batch = False )
	trainer.set_save_steps(0,'./checkpoint/')
	trainer.set_early_stop(early_stop)
	trainer.set_lr_decay(0.0)
	trainer.set_weight_decay(0.0)
	trainer.run()
	if not early_stop:
		ensemble_model.save_checkpoint('./checkpoint/AReWE_Embeddings_%s.ckpt'%data_folder)
		embeddings = ensemble_model.get_parameters("numpy")
		np.save("embeddings/AReWE_Embeddings_%s.npy"%data_folder,embeddings)
		print ("The model parameters have been saved correctly...")

link_prediction='micro' #or 'macro'

TEST_FILES = {
    "FB15K237": "test2id.txt",
    "WN18RR": "test2id.txt",
    "NELL-995": "test2id_refined.txt",
}

ensemble_model.load_checkpoint('./checkpoint/AReWE_Embeddings_%s.ckpt'%data_folder)
# ensemble_model.load_checkpoint('./AReW_Ens_FineTuned/%s_FullModel_MRR.ckpt'%data_folder)

if link_prediction=='micro': # test the model (micro)
	test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,TEST_FILES[data_folder], "link")
	tester1 = Tester(model = ensemble_model, data_loader = test_dataloader_all, use_gpu = True)
	tester1.run_link_prediction(type_constrain = True)

elif link_prediction=='macro': # test the model (macro)
	
	rel_dic={}
	with open("./benchmarks/%s/valid2id.txt"%data_folder) as valid: 
		for line in valid: 
			line=line.strip().split()
			if len(line)!=1:
				h,t,r = int(line[0]),int(line[1]),int(line[2])
				rel_dic.setdefault(r,set())
				rel_dic[r].add((h,t))
	list_=[]
	for rel in rel_dic:
		list_.append((len(rel_dic[rel]),rel))
	list_.sort(reverse=True)
	ordered_rel=[x[1] for x in list_]
	ind=int((len(ordered_rel)*50)/100)
	tail_relations = ordered_rel[-ind:]
	head_relaitons = ordered_rel[:50]
	body_relations=ordered_rel[50:110]
	print(len(tail_relations)+len(head_relaitons)+len(body_relations))

	test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,TEST_FILES[data_folder], "link")
	tester1 = Tester(model = metaembed, data_loader = test_dataloader_all, use_gpu = True)
	MRR=[]
	MR=[]
	Hits1=[]
	Hits3=[]
	Hits10=[]
	All_Relation=[i for i in range(len(ordered_rel))]

	for rel in All_Relation:
		print("Relation",rel)
		mrr, mr, hit10, hit3, hit1,left_mrr,right_mrr=tester1.run_link_prediction(relation=rel,type_constrain = False)
		# if not np.isnan(mrr):
		MRR.append(mrr)
		MR.append(mr)
		Hits1.append(hit1)
		Hits3.append(hit3)
		Hits10.append(hit10)
	print("Macro-average Results..")
	print("MRR=%f, MR=%f, Hits@10=%f, Hists@3=%f, Hits@1=%f"%(np.mean(MRR),np.mean(MR),np.mean(Hits10),np.mean(Hits3),np.mean(Hits1)))





		
