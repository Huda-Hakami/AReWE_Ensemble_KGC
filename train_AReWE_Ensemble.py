from openke.config import TrainerAReWE, Tester
from openke.module.model import OptimizedEnsemble
from openke.module.loss import  MarginLossAReWE
from openke.module.strategy import NegativeSamplingAReWE
from openke.data import PyTorchTrainDataLoader, TestDataLoader
import numpy as np 
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Train AReWE")
	parser.add_argument("--data_folder",type=str,default='FB15K237',required=True,help="Dataset folder, e.g., FB15K237, WN18RR, or NELL-995")
	parser.add_argument("--batch_size",type=int,default=1024,help="Training batch size")
	parser.add_argument("--epochs",type=int,default=50,help="Number of training epochs")
	parser.add_argument("--lr",type=float,default=0.0001,help="Learning rate")
	parser.add_argument("--margin",type=float,default=1.0,help="margin")
	parser.add_argument("--weighting_strategy",type=str,default="score_based",choices=["score_based", "mrr_based"],help="Relation-specific weighting strategy")
	parser.add_argument("--negEnt",type=int,default=200,help="Number of negative examples per positive triple")
	parser.add_argument("--train_split",type=str,default="valid",choices=["train", "valid"],help="Data split used to fine-tune AReWE")
	parser.add_argument("--do_train",type=lambda x: x.lower() == "true",default=True,help="Whether to fine-tune AReWE (True) or only test (False).")
	parser.add_argument("--TCNS",type=lambda x: x.lower() == "true",default=False,help="Whether to apply type constraint negative sampling.")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.train_split=='train':
		tri_file='train2id.txt'
	elif args.train_split=='valid':
		tri_file='valid2id.txt'
	
	train_dataloader = PyTorchTrainDataLoader(
		in_path = "./benchmarks/%s/"%args.data_folder, 
		tri_file = tri_file,
		batch_size= args.batch_size,
		threads = 1, 
		sampling_mode = 'cross, 
		bern_flag = 1,
		TC_NS = args.TCNS, 
		filter_flag = 1, 
		neg_ent = args.negEnt,
		neg_rel = 0
	)
	
	print ("the model is defined correctly...")

	# define the ensemble model
	ensemble_model = OptimizedEnsemble(
		data_folder = args.data_folder,
		batch_size = train_dataloader.get_batch_size(),
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		neg_examples=args.negEnt
	)

	early_stop=0
	if args.do_train:  # Fine-tuning AReWE then testing
		# train the model
		model = NegativeSamplingAReWE(
			model = ensemble_model, 
			loss = MarginLossAReWE(margin = args.margin),
			weighting_strategy = args.weighting_strategy,  
			regul_rate1 = 0.0, #l2 reg
			regul_rate2 = 0.0, #orthogonality 
		)

		trainer = TrainerAReWE(model = model, data_loader = train_dataloader, train_times = args.epochs, alpha = args.lr, use_gpu = True, opt_method = "adam", data_folder=args.data_folder, negEnt=args.negEnt, rel_batch = False )
		trainer.set_save_steps(0,'./checkpoint/')
		trainer.set_early_stop(early_stop)
		trainer.set_lr_decay(0.0)
		trainer.set_weight_decay(0.0)
		trainer.run()
		if not early_stop:
			ensemble_model.save_checkpoint('./checkpoint/AReWE_Embeddings_%s.ckpt'%args.data_folder)
			embeddings = ensemble_model.get_parameters("numpy")
			np.save("embeddings/AReWE_Embeddings_%s.npy"%data_folder,embeddings)
			print ("The model parameters have been saved correctly...")
	else:  # Testing
		ensemble_model.load_checkpoint('./AReW_Ens_FineTuned/%s_FullModel_SumScores_Valid.ckpt'%args.data_folder)


	link_prediction='micro' #or 'macro'

	TEST_FILES = {
		"FB15K237": "test2id.txt",
		"WN18RR": "test2id.txt",
		"NELL-995": "test2id_refined.txt",
	}

	# ensemble_model.load_checkpoint('./checkpoint/AReWE_Embeddings_%s.ckpt'%args.data_folder)
	# # ensemble_model.load_checkpoint('./AReW_Ens_FineTuned/%s_FullModel_MRR.ckpt'%data_folder)

	if link_prediction=='micro': # test the model (micro)
		test_dataloader_all = TestDataLoader("./benchmarks/%s/"%args.data_folder,TEST_FILES[args.data_folder], "link")
		tester1 = Tester(model = ensemble_model, data_loader = test_dataloader_all, use_gpu = True)
		tester1.run_link_prediction(type_constrain = True)

	elif link_prediction=='macro': # test the model (macro)
		
		rel_dic={}
		with open("./benchmarks/%s/valid2id.txt"%args.data_folder) as valid: 
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

		test_dataloader_all = TestDataLoader("./benchmarks/%s/"%args.data_folder,TEST_FILES[args.data_folder], "link")
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
