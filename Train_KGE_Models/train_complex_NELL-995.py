import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import sys 
import os 
import numpy as np 
os.environ['CUDA_VISIBLE_DEVICES']='1'

data_folder="NELL-995"
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/%s/"%data_folder, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 50,
	neg_rel = 0
)


# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300
)

# define the loss function
model = NegativeSampling(
	model = complEx, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate1 = 0.1
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 4000, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
complEx.save_checkpoint('./checkpoint/complEx.ckpt')

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
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



embeddings = complEx.get_parameters("numpy")
np.save("embeddings/complex_embeddings_%s.npy"%data_folder,embeddings)



