import sys
import os
sys.path.append(os.path.abspath(".."))

import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np
data_folder="WN18RR"
train=True 

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../benchmarks/%s/"%data_folder,
	batch_size = 1000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("../benchmarks/WN18RR/", "link")

# define the model
rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300,
	margin = 6.0,
	epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = 0.5),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
if train: 
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1500, alpha = 2e-5, use_gpu = True, opt_method = "adam")
	trainer.run()
	rotate.save_checkpoint('../checkpoint/rotate_%s.ckpt'%data_folder)
	embeddings = rotate.get_parameters("numpy")
	np.save("../embeddings/rotate_embeddings_%s.npy"%data_folder,embeddings)
	print ("The model parameters have been saved correctly...")

# test the model
rotate.load_checkpoint('../checkpoint/rotate_%s.ckpt'%data_folder)
print ("==========================================")
print ("TEST_All")
sys.stdout.flush()
# dataloader for test
test_dataloader_all = TestDataLoader("../benchmarks/%s/"%data_folder,"test2id.txt", "link")
tester1 = Tester(model = rotate, data_loader = test_dataloader_all, use_gpu = True)
tester1.run_link_prediction(type_constrain = True)
tester1.run_relation_prediction(type_constrain = False)
acc,thre=tester1.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_all
del tester1

