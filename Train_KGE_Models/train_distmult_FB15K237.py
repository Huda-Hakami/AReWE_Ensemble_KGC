import sys
import os
sys.path.append(os.path.abspath(".."))

import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader 
import numpy as np 

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

print ("DistMult Model")
data_folder="FB15K237"
negEnt=1
negRel=0
sys.stdout.flush()

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../benchmarks/%s/"%data_folder, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 0, 
	neg_ent = negEnt,
	neg_rel = negRel
)


# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300
)

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.05
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 10, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
distmult.save_checkpoint('../checkpoint/distmult.ckpt')

# test the model
distmult.load_checkpoint('../checkpoint/distmult.ckpt')
print ("==========================================")
print ("TEST_All")
sys.stdout.flush()
# dataloader for test
test_dataloader_all = TestDataLoader("../benchmarks/%s/"%data_folder,"test2id.txt", "link")
tester1 = Tester(model = distmult, data_loader = test_dataloader_all, use_gpu = True)
tester1.run_link_prediction(type_constrain = True)
tester1.run_relation_prediction(type_constrain = False)
acc,thre=tester1.run_triple_classification()
print (acc,thre)
sys.stdout.flush()

embeddings = distmult.get_parameters("numpy")
np.save("../embeddings/distmult_embeddings_negent%d_negrel%d_%s.npy"%(negEnt,negRel,data_folder),embeddings)
