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

data_folder="FB15K237"
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/%s/"%data_folder, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)


# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100
)

# define the loss function
model = NegativeSampling(
	model = complEx, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 50, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
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

print ("==========================================")
print ("TEST_WithMentions")
sys.stdout.flush()
test_dataloader_withMentions = TestDataLoader("./benchmarks/%s/"%data_folder,"test2id_withMentions.txt", "link")
tester2 = Tester(model = complEx, data_loader = test_dataloader_withMentions, use_gpu = True)
tester2.run_link_prediction(type_constrain = True)
tester2.run_relation_prediction(type_constrain = False)
acc,thre=tester2.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_withMentions
del tester2

print ("==========================================")
print ("TEST_WithoutMentions")
sys.stdout.flush()
test_dataloader_withoutMentions = TestDataLoader("./benchmarks/%s/"%data_folder, "test2id_withoutMentions.txt",  "link")
tester3 = Tester(model = complEx, data_loader = test_dataloader_withoutMentions, use_gpu = True)
tester3.run_link_prediction(type_constrain = True)
tester3.run_relation_prediction(type_constrain = False)
acc,thre=tester3.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_withoutMentions
del tester3

print ("==========================================")
print ("TEST_1-1")
sys.stdout.flush()
test_dataloader_1to1 = TestDataLoader("./benchmarks/%s/"%data_folder, "1-1.txt",  "link")
tester4 = Tester(model = complEx, data_loader = test_dataloader_1to1, use_gpu = True)
tester4.run_link_prediction(type_constrain = True)
sys.stdout.flush()
del test_dataloader_1to1
del tester4

print ("==========================================")
print ("TEST_1-n")
sys.stdout.flush()
test_dataloader_1ton = TestDataLoader("./benchmarks/%s/"%data_folder, "1-n.txt",  "link")
tester5 = Tester(model = complEx, data_loader = test_dataloader_1ton, use_gpu = True)
tester5.run_link_prediction(type_constrain = True)
sys.stdout.flush()
del test_dataloader_1ton
del tester5

print ("==========================================")
print ("TEST_n-1")
sys.stdout.flush()
test_dataloader_nto1 = TestDataLoader("./benchmarks/%s/"%data_folder, "n-1.txt",  "link")
tester6 = Tester(model = complEx, data_loader = test_dataloader_nto1, use_gpu = True)
tester6.run_link_prediction(type_constrain = True)
sys.stdout.flush()
del test_dataloader_nto1
del tester6

print ("==========================================")
print ("TEST_n-n")
sys.stdout.flush()
test_dataloader_nton = TestDataLoader("./benchmarks/%s/"%data_folder, "n-n.txt",  "link")
tester7 = Tester(model = complEx, data_loader = test_dataloader_nton, use_gpu = True)
tester7.run_link_prediction(type_constrain = True)
sys.stdout.flush()
del test_dataloader_nton
del tester7

embeddings = complEx.get_parameters("numpy")
np.save("embeddings/complex_embeddings_%s.npy"%data_folder,embeddings)



