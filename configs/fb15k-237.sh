#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/FB15K-237/"
vocab_dir="datasets/data_preprocessed/FB15K-237/vocab"
max_policy_high=4
max_policy_low=2
reward_gamma=1.2
total_iterations=2000
path_length=3
hidden_size=50
embedding_size=100
relation_size=100
batch_size=100
cluster=80
regularization_lamda=0.005
beta=0.02
Lambda=0.02
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/fb15k-237/"
load_model=0
model_load_dir="saved_models/fb15k-237"
model_load_dir="/home/gjwang/RLH/fb15k-237/log/rlhmodel/model.ckpt"
nell_evaluation=0
