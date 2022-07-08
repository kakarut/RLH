# RLH
Code and models for the paper [Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning]

Inspired by the hierarchical reasoning principle of human cognitive decision-making, the model RLH based on hierarchical reinforcement
learning is proposed to solve the multi-semantic problem of knowledge graph multi-hop reasoning process.


## Requirements
To install the various python 3.0
```
pip install -r requirements.txt
```

## Training

You can get the hyperparam configs from the directory **configs**.
```
sh run.sh configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the dateset file, e.g., WN18RR.sh

## Testing
make
```
model_load_dir="saved_models/WN18RR/model.ckpt"
```
## Citation
If you use this code, please cite the paper
```
@inproceedings{DBLP:conf/ijcai/WanP00H20,
  title={Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning},
  author={Guojia Wan and Shirui Pan and Chen Gong and Chuan Zhou and Gholamreza Haffari},
  booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI} 2020},
  pages={1926--1932},
  year={2020},
}
```
