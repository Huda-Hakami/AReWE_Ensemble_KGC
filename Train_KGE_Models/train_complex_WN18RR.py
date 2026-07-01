import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np  
import sys 
print ("DistMult Model")
data_folder="WN18RR"
negEnt=25
negRel=0
train=True

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/%s/"%data_folder, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = negEnt,
	neg_rel = negRel
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = complEx, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate1 = 1.0
)

# train the model
if train: 
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
	trainer.run()
	complEx.save_checkpoint('./checkpoint/complex_%s.ckpt'%data_folder)
	embeddings = complEx.get_parameters("numpy")
	np.save("embeddings/complex_embeddings_%s.npy"%data_folder,embeddings)
	print ("The model parameters have been saved correctly...")

# test the model
complEx.load_checkpoint('./checkpoint/complex_%s.ckpt'%data_folder)
print ("==========================================")
print ("TEST_All")
sys.stdout.flush()
# dataloader for test
test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,"test2id.txt", "link")
tester1 = Tester(model = complEx, data_loader = test_dataloader_all, use_gpu = True)
tester1.run_link_prediction(type_constrain = True)
tester1.run_relation_prediction(type_constrain = False)
acc,thre=tester1.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_all
del tester1
