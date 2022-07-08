# RLH
Code and models for the paper [Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning]

We development the code based on the code of MINERVA [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning] (https://github.com/shehzaadzd/MINERVA)

Inspired by the hierarchical reasoning principle of human cognitive decision-making, the model RLH based on hierarchical reinforcement
learning is proposed to solve the multi-semantic problem of knowledge graph multi-hop reasoning process.


## Requirements
To install the various python 3.0
```
pip install -r requirements.txt
```

## Training
The hyperparam configs for each experiments are in the **configs** directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the config file. For example, 
```
sh run.sh configs/WN18RR.sh
```

## Testing
make
```
load_model=1
model_load_dir="saved_models/WN18RR/model.ckpt"
```


## Citation
If you use this code, please cite the paper
```
@inproceedings{li2018path,
  title={Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning},
  author={Guojia Wan and Shirui Pan and Chen Gong and Chuan Zhou and Gholamreza Haffari},
  booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI} 2020},
  pages={1926--1932},
  year={2020},
}
```
