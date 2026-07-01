# AReWE: Adaptive Relation-specific Ensemble Learning for Knowledge Graph Completion

This repository contains the reference implementation of AReWE (Adaptive Relation-specific Ensemble), a relation-aware ensemble framework for Knowledge Graph Completion (KGC). AReWE jointly fine-tunes multiple Knowledge Graph Embedding (KGE) models and adaptively estimates relation-specific weights during training to improve link prediction performance.
The implementation supports training, evaluation, and reproduction of the experiments reported in the accompanying paper.

## Features
* Joint fine-tuning of multiple KGE models (TransE, DistMult, ComplEx, RotatE and ConvKB)
* Support for the benchmark datasets:
    * FB15K237
    * WN18RR
    * NELL-995

## Repository Structure

```text
.
├── benchmarks/                 # Benchmark datasets
├── checkpoints/                # Saved model checkpoints
├── PreTrained_KGEs/            # Pretrained embeddings for individual KGE models
├── AReWE_Ens_FineTuned/        # Finetuned AReWE ensemble embeddings
├── openke/                     # Modified OpenKE implementation
├── train_AReWE_Ensemble.py     # Training and evaluation script
├── Train_KGE_Models/           # Code to train individual KGE models
└── README.md
```



## Datasets
Each dataset should contain the corresponding training, validation, and testing files required by OpenKE.

Training
Configure the desired dataset and training hyperparameters in train.py.
Run training using:
python3 train_AReWE_Ensemble.py
The trained model checkpoint will be saved in:
checkpoints/
Learned embeddings will be stored in:
embeddings/


## Evaluation
The evaluation mode (micro or macro) can be selected through the corresponding configuration.
Reported metrics include:
* Mean Reciprocal Rank (MRR)
* Mean Rank (MR)
* Hits@1
* Hits@3
* Hits@10


## Acknowledgements
This repository builds upon the OpenKE-PyTorch framework for Knowledge Graph Embedding (KGE) models.

For more information about OpenKE, please visit the official project website: 
[OpenKE Project Website](http://139.129.163.161//).
[OpenKE GitHub Repository](https://github.com/thunlp/OpenKE).

## Citation
If you use this implementation in your research, please cite the accompanying paper. The citation will be added after publication.
@article{anonymous2026,
  title={Anonymous Submission},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}

License
This project is released under the MIT License (or the license included in this repository).
