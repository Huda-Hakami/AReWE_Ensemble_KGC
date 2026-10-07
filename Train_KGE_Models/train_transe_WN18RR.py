import sys
import os
sys.path.append(os.path.abspath(".."))

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np 

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

print ("DistMult Model")
data_folder="WN18RR"
negEnt=25
negRel=0
Train=True
sys.stdout.flush()

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../benchmarks/%s/"%data_folder,
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = negEnt,
	neg_rel = negRel)



# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 50, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
if Train:
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 3000, alpha = 0.0005, use_gpu = True, opt_method = "adagrad")
	trainer.run()
	transe.save_checkpoint('../checkpoint/transe_%s.ckpt'%data_folder)

	embeddings = transe.get_parameters("numpy")
	np.save("../embeddings/transe_embeddings_%s.npy"%data_folder,embeddings)
	print("parameters are saved ...")
	sys.stdout.flush()

# test the model
transe.load_checkpoint('../checkpoint/transe_%s.ckpt'%data_folder)
print ("==========================================")
print ("TEST_All")
sys.stdout.flush()
# dataloader for test
test_dataloader_all = TestDataLoader("../benchmarks/%s/"%data_folder,"test2id.txt", "link")
tester1 = Tester(model = transe, data_loader = test_dataloader_all, use_gpu = True)
tester1.run_link_prediction(type_constrain = True)
tester1.run_relation_prediction(type_constrain = False)
acc,thre=tester1.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_all
del tester1





