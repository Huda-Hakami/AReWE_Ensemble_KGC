import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import sys 
import numpy as np 

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
	neg_rel = 0 # neg_rel = 0 which mean that we don't corrupt the relation
)


# define the model
rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300,
	margin = 9.0,#jjjjjjjj it was 9 i want to try 2
	epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = 1),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate1 = 0.0
)

# train the model (train_time=1500,alpha=0.5)
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 2e-5, use_gpu = True, opt_method = "adam")
trainer.run()
rotate.save_checkpoint('./checkpoint/rotate.ckpt')

# test the model
rotate.load_checkpoint('./checkpoint/rotate.ckpt')
print ("==========================================")
print ("TEST_All")
sys.stdout.flush()
# dataloader for test
test_dataloader_all = TestDataLoader("./benchmarks/%s/"%data_folder,"test2id.txt", "link")
tester1 = Tester(model = rotate, data_loader = test_dataloader_all, use_gpu = True)
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
tester2 = Tester(model = rotate, data_loader = test_dataloader_withMentions, use_gpu = True)
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
tester3 = Tester(model = rotate, data_loader = test_dataloader_withoutMentions, use_gpu = True)
tester3.run_link_prediction(type_constrain = True)
tester3.run_relation_prediction(type_constrain = False)
acc,thre=tester3.run_triple_classification()
print (acc,thre)
sys.stdout.flush()
del test_dataloader_withoutMentions
del tester3

print ("==========================================")
print ("TEST_1-n")
sys.stdout.flush()
test_dataloader_1ton = TestDataLoader("./benchmarks/%s/"%data_folder, "1-n-ver2.txt",  "link")
tester4 = Tester(model = rotate, data_loader = test_dataloader_1ton, use_gpu = True)
tester4.run_link_prediction(type_constrain = True)
sys.stdout.flush()
del test_dataloader_1ton
del tester4

print ("==========================================")
print ("TEST_n-1")
sys.stdout.flush()
test_dataloader_nto1 = TestDataLoader("./benchmarks/%s/"%data_folder, "n-1-ver2.txt",  "link")
tester5 = Tester(model = rotate, data_loader = test_dataloader_nto1, use_gpu = True)
tester5.run_link_prediction(type_constrain = True)
sys.stdout.flush()
del test_dataloader_nto1
del tester5

embeddings = rotate.get_parameters("numpy")
np.save("embeddings/rotate_embeddings_%s.npy"%data_folder,embeddings)

