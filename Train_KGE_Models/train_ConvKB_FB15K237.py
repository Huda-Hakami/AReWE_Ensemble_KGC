import sys
import os
sys.path.append(os.path.abspath(".."))

import openke
from openke.config import Trainer, Tester
from openke.module.model import ConvKB
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader,PyTorchTrainDataLoader
import numpy as np 
data_folder="FB15K237"
train=True
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../benchmarks/%s/"%data_folder,
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", #it was normal, I am checking where this make better balance between head/tail prediction! 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 10,
	neg_rel = 0)

# train_dataloader = PyTorchTrainDataLoader(
# 	in_path = "./benchmarks/%s/"%data_folder, 
# 	nbatches = 100,
# 	threads = 8, 
# 	sampling_mode = "normal", 
# 	bern_flag = 1, 
# 	filter_flag = 1, 
# 	neg_ent = 10,
# 	neg_rel = 0
# )



# define the model
convkb = ConvKB(
	data_folder = data_folder,
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100,
	out_channels=128,
	use_init_embeddings=True)


# define the loss function
model = NegativeSampling(
	model = convkb, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(),
	regul_rate = 0.1
)

# train the model
if train:
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 400, alpha = 0.01, use_gpu = True, opt_method = "adagrad",data_folder=data_folder)
	trainer.set_save_steps(0,'./checkpoint/')
	trainer.run()
	convkb.save_checkpoint('./checkpoint/convkb_%s.ckpt'%data_folder)

	embeddings = convkb.get_parameters("numpy")
	np.save("embeddings/convkb_embeddings_%s.npy"%data_folder,embeddings)
	print ("The model parameters have been saved correctly...")

# test the model
convkb.load_checkpoint('./checkpoint/convkb_%s.ckpt'%data_folder)
print ("==========================================")
print ("TEST_All")
sys.stdout.flush()
# dataloader for test
test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,"test2id.txt", "link")
tester1 = Tester(model = convkb, data_loader = test_dataloader_all, use_gpu = True)
tester1.run_link_prediction(type_constrain = True)
tester1.run_relation_prediction(type_constrain = False)
acc,thre=tester1.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_all
del tester1






