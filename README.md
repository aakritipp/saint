This repository is the official PyTorch implementation of RAFD. Find the paper on (https://doi.org/10.1145/3677052.3698692)

# Retrieval Augmented Fraud Detection (RAFD)


![Overview](pipelineRAFD.png)

This repository implements a novel approach to financial fraud detection by combining the power of SAINT (Self-Attention and Intersection Neural Network with Transformers) with retrieval-augmented classification techniques. The model addresses critical challenges in fraud detection, particularly the extreme class imbalance and evolving nature of fraudulent patterns.
Architecture
RAFD consists of four main components:

Encoder Module: Utilizes a pre-trained SAINT model as the backbone to generate embeddings for both input samples and training data. SAINT's self-attention mechanisms and contrastive pre-training make it particularly effective for tabular data.

Retrieval Module: Implements efficient similarity search using FAISS (Facebook AI Similarity Search) to find relevant context samples for each input transaction. This module helps enrich the representation of minority class samples.
Integration Module: Combines input embeddings with retrieved context through:

Integration Module: Similarity computation using squared Euclidean distance
Value computation incorporating both feature differences and label information
Weighted aggregation of context information


Predictor Module: A three-layer MLP architecture that makes the final fraud prediction based on the enriched representation.

## Requirements

We recommend using `anaconda` or `miniconda` for python. Our code has been tested with `python=3.8` on linux.

Create a conda environment from the yml file and activate it.
```
conda env create -f rafd_environment.yml
conda activate rafd_env
```

Make sure the following requirements are met

* torch>=1.8.1
* torchvision>=0.9.1

### Optional
We used wandb to update our logs. But it is optional.
```
conda install -c conda-forge wandb 
```


## Training & Evaluation

In each of our experiments, we use a single Tesla T4 16GB GPU.


To train the model(s) in the paper, run this command:

```
python train_RAC.py --dset_id <openml_dataset_id> --savemodelroot <pretrained model> --task <task_name> --attentiontype <attention_type> --cont_embeddings <embedding size> --context_size <context size>
```



### Arguments
* `--dset_id` : Dataset id from OpenML. Works with all the datasets mentioned in the paper. Works with all OpenML datasets.
* `--task` : The task we want to perform. Pick from 'regression','multiclass', or 'binary'.
* `--attentiontype` : Variant of SAINT. 'col' refers to SAINT-s variant, 'row' is SAINT-i, and 'colrow' refers to SAINT.
* `--embedding_size` : Size of the feature embeddings
* `--transformer_depth` : Depth of the model. Number of stages.
* `--attention_heads` : Number of attention heads in each Attention layer.
* `--cont_embeddings` : Style of embedding continuous data.
* `--pretrain` : To enable pretraining
* `--pt_tasks` : Losses we want to use for pretraining. Multiple arguments can be passed.
* `--pt_aug` : Types of data augmentations used in pretraining. Multiple arguments are allowed. We support only mixup and CutMix right now.
* `--ssl_samples` : Number of labeled samples used in semi-supervised experiments. 
* `--pt_projhead_style` : Projection head style used in contrastive pipeline.
* `--nce_temp` : Temperature used in contrastive loss function.
* `--active_log` : To update the logs onto wandb. This is optional

#### <span style="color:Tomato">Most of the hyperparameters are hardcoded in train.py file. For datasets with really high number of features, we suggest using smaller batchsize, lower embedding dimension and fewer number of heads.</span>


## Data Preparation
The model supports two main datasets:

European Credit Card Default Dataset\
IEEE-CIS Fraud Detection Dataset

Data should be organized as follows:
```
Copydata/
├── european_credit/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── ieee_cis/
    ├── train.csv
    ├── valid.csv
    └── test.csv
```
```    
Training
bashCopypython train_rac.py \
    --dset_id 1 \
    --task binary \
    --embedding_size 32 \
    --context_size 60 \
    --epochs 25 \
    --batchsize 256
```

Key arguments:

context_size: Number of similar samples to retrieve (recommended: 60-120)\
embedding_size: Dimension of SAINT embeddings (default = 32)


## Performance
On benchmark datasets, RAFD achieves:

European Credit Card Dataset: 0.833 AUCPR (2.209% improvement over SAINT)
IEEE CIS Dataset: 0.557 AUCPR (1.089% improvement over SAINT)

## Key Features

Dynamic Context Enhancement: Automatically enriches minority class representations through relevant sample retrieval
Efficient Similarity Search: Uses FAISS for fast and scalable nearest neighbor search
Flexible Architecture: Can work with different encoder backbones (demonstrated with SAINT)
Interpretable Results: Retrieved samples provide insights into model decisions


## Citations and References

```
This implementation is based on the methodology described in the following paper:
RAFD:
@inproceedings{10.1145/3677052.3698692,
    author = {Pandey, Anubha},
    title = {Retrieval Augmented Fraud Detection},
    year = {2024}, isbn = {9798400710810},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3677052.3698692},
    doi = {10.1145/3677052.3698692}
}

SAINT:
@article{somepalli2021saint,
  title={SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training},
  author={Somepalli, Gowthami and Goldblum, Micah and Schwarzschild, Avi and Bruss, C Bayan and Goldstein, Tom},
  journal={arXiv preprint arXiv:2106.01342},
  year={2021}
}
```
