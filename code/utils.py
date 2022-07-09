import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab
 from __future__ import division
from __future__ import print_function
import random
from collections import namedtuple, Counter

from BFS.KB import KB
from BFS.BFS import BFS

matplotlib.use('Agg')



def loadData(str):
    fr = open(str)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    matA = mat(datArr)
    print(matA.shape)
    nameArr = [line[0] for line in sArr]
    return matA, nameArr

def pca(inputM, k):
    meansVals = mean(inputM, axis =0)
    inputM = inputM - meansVals
    covM = cov(inputM, rowvar=0)
    s, V = linalg.eig(covM) 
    paixu = argsort(s) 
    paixuk = paixu[:-(k+1):-1] 
    kwei = V[:,paixuk]
    outputM = inputM * kwei 
    chonggou = (outputM * kwei.T)+meansVals
    return outputM,chonggou
    #return chonggou,outputM

def plotV(a, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    font = { 'fontname':'Liberation Mono', 'fontsize':0.5, 'verticalalignment': 'top', 'horizontalalignment':'center' }
    ax.scatter(a[:,0].tolist(), a[:,1].tolist(), marker = ' ')
    ax.set_xlim(-0.8,0.8)
    ax.set_ylim(-0.8,0.8)
    i = 0
    for label, x, y in zip(labels, a[:, 0], a[:, 1]):
        i += 1
        s = random.uniform(0,100)
        if i<14951:
            if s > 3.1:
                continue
        else:
            if s > 6.7:
                continue
        ax.annotate(label, xy = (x, y), xytext = None, ha = 'right', va = 'bottom', **font)


    plt.title('TransE pca2dim')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('plot_with_labels2', dpi = 3000, bbox_inches = 'tight' ,orientation = 'landscape', papertype = 'a0')


  
def distance(data, centers):
    
    centers = centers.astype(np.float64)
    dist = np.zeros((data.shape[0], centers.shape[0])) 
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i, j] = np.sqrt(np.sum((data.iloc[i, :] - centers[j]) ** 2)) 
                                                                           
    dist = dist.astype(np.float64)
    return dist

def near_center(data, centers): 
    dist = distance(data, centers)
    dist = dist.astype(np.float64)
    near_cen = np.argmin(dist, 1) 
    near_cen = near_cen.astype(np.float64)
    return near_cen

def kmeans(data, k):
    centers = np.random.choice(np.arange(-5, 5, 0.1), (k, 2)) 
    print(centers)

    for _ in range(10): 
        near_cen = near_center(data, centers)
       
        for ci in range(k): 
            centers[ci] = data[near_cen == ci].mean()

    centers = centers.astype(np.float64)
    near_cen = near_cen.astype(np.float64)

    return centers, near_cen

centers, near_cen = kmeans(data, 40)
plt.subplot(1,1,1)
plt.scatter(x, y, c=near_cen)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=25, c='r')
plt.savefig('kmeans', dpi = 3000, bbox_inches = 'tight' ,orientation = 'landscape', papertype = 'a0')
 


# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = '../NELL-995/'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))

def compare(v1, v2):
    return sum(v1 == v2)

def teacher(e1, e2, num_paths, env, path = None):
	f = open(path)
	content = f.readlines()
	f.close()
	kb = KB()
	for line in content:
		ent1, rel, ent2 = line.rsplit()
		kb.addRelation(ent1, rel, ent2)
	# kb.removePath(e1, e2)
	intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
	res_entity_lists = []
	res_path_lists = []
	for i in range(num_paths):
		suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
		suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
		if suc1 and suc2:
			res_entity_lists.append(entity_list1 + entity_list2[1:])
			res_path_lists.append(path_list1 + path_list2)
	print('BFS found paths:', len(res_path_lists))
	
	# ---------- clean the path --------
	res_entity_lists_new = []
	res_path_lists_new = []
	for entities, relations in zip(res_entity_lists, res_path_lists):
		rel_ents = []
		for i in range(len(entities)+len(relations)):
			if i%2 == 0:
				rel_ents.append(entities[int(i/2)])
			else:
				rel_ents.append(relations[int(i/2)])

		#print(rel_ents)

		entity_stats = Counter(entities).items()
		duplicate_ents = [item for item in entity_stats if item[1]!=1]
		duplicate_ents.sort(key = lambda x:x[1], reverse=True)
		for item in duplicate_ents:
			ent = item[0]
			ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
			if len(ent_idx)!=0:
				min_idx = min(ent_idx)
				max_idx = max(ent_idx)
				if min_idx!=max_idx:
					rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
		entities_new = []
		relations_new = []
		for idx, item in enumerate(rel_ents):
			if idx%2 == 0:
				entities_new.append(item)
			else:
				relations_new.append(item)
		res_entity_lists_new.append(entities_new)
		res_path_lists_new.append(relations_new)
	
	print(res_entity_lists_new)
	print(res_path_lists_new)

	good_episodes = []
	targetID = env.entity2id_[e2]
	for path in zip(res_entity_lists_new, res_path_lists_new):
		good_episode = []
		for i in range(len(path[0]) -1):
			currID = env.entity2id_[path[0][i]]
			nextID = env.entity2id_[path[0][i+1]]
			state_curr = [currID, targetID, 0]
			state_next = [nextID, targetID, 0]
			actionID = env.relation2id_[path[1][i]]
			good_episode.append(Transition(state = env.idx_state(state_curr), action = actionID, next_state = env.idx_state(state_next), reward = 1))
		good_episodes.append(good_episode)
	return good_episodes

def path_clean(path):
	rel_ents = path.split(' -> ')
	relations = []
	entities = []
	for idx, item in enumerate(rel_ents):
		if idx%2 == 0:
			relations.append(item)
		else:
			entities.append(item)
	entity_stats = Counter(entities).items()
	duplicate_ents = [item for item in entity_stats if item[1]!=1]
	duplicate_ents.sort(key = lambda x:x[1], reverse=True)
	for item in duplicate_ents:
		ent = item[0]
		ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
		if len(ent_idx)!=0:
			min_idx = min(ent_idx)
			max_idx = max(ent_idx)
			if min_idx!=max_idx:
				rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
	return ' -> '.join(rel_ents)

def prob_norm(probs):
	return probs/sum(probs)

if __name__ == '__main__':
	print(prob_norm(np.array([1,1,1])))
