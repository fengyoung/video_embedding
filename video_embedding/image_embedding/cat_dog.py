import sys
import os
import numpy as np


def cosine_sim(v1, v2):
	if len(v1) != len(v2):
		return False
	cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	return 0.5 + 0.5 * cos

def euclidean_sim(v1, v2):
	if len(v1) != len(v2):
		return False
	euc_dist = np.linalg.norm(v1 - v2)
	return 1.0 / (1.0 + euc_dist)


def read_features(image_feat_file):
	image_feats = []	
	fp = open(image_feat_file, 'r')
	line = fp.readline()
	while line:
		if len(line.rstrip()) > 0:
			ss = line.rstrip().split('\t')
			feats = map(lambda x: float(x), ss[2].split(','))				
			image_feats.append([ss[0], ss[1], feats])
		line = fp.readline()
	fp.close()
	return image_feats


image_feats = read_features("./model/02.feats")
train_feats = read_features("./model/01_train_20170419.feats")

v_cat2 = np.array(image_feats[0][2])
v_cat1dog1 = np.array(image_feats[1][2])
v_cat1 = np.array(image_feats[2][2])
v_dog1 = np.array(image_feats[3][2])
v_dog2 = np.array(image_feats[4][2])
v1 = v_cat1dog1 - v_dog1
v2 = v_cat1dog1 - v_dog2


print("sim(v1, v_cat1) = %.6g" % cosine_sim(v1, v_cat1))
print("sim(v1, v_cat2) = %.6g" % cosine_sim(v1, v_cat2))
print("sim(v2, v_cat1) = %.6g" % cosine_sim(v2, v_cat1))
print("sim(v2, v_cat2) = %.6g" % cosine_sim(v2, v_cat2))


v_query = v_cat1dog1
#v_query = v1
#v_query = v2
#v_query = v_cat1
#v_query = v_cat2

sim_vals = []	
for tfeat in train_feats:
	sim = cosine_sim(v_query, np.array(tfeat[2]))
	sim_vals.append([tfeat[0], tfeat[1], sim])
sim_vals = sorted(sim_vals, cmp = lambda x, y: cmp(x[2], y[2]), reverse = True)
topn = np.min([10, len(sim_vals)])
for t in range(topn): 
	print("top %d | sim: %.6g | id: %s, file: %s" % (t+1, sim_vals[t][2], sim_vals[t][0], sim_vals[t][1]))
tt = range(topn)
tt.reverse()
for t in tt:
	os.system('open %s' % sim_vals[t][1])





