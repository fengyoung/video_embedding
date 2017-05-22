import sys
sys.path.append("../")

import os
import numpy as np
import image_retrieval


def cat_dog_demo(feats_test_file, feats_train_file):
	test_feats = image_retrieval.read_features(feats_test_file)
	train_feats = image_retrieval.read_features(feats_train_file)
	
	v_cat2 = np.array(test_feats[0][2])
	v_cat1dog1 = np.array(test_feats[1][2])
	v_cat1 = np.array(test_feats[2][2])
	v_dog1 = np.array(test_feats[3][2])
	v_dog2 = np.array(test_feats[4][2])
	v1 = v_cat1dog1 - v_dog1
	v2 = v_cat1dog1 - v_dog2

	v_query = v_cat1dog1
#	v_query = v1
#	v_query = v2
#	v_query = v_cat1
#	v_query = v_cat2
	
	sim_vals = image_retrieval_by_feat(v_query, test_feats, sim_type): 
	topn = np.min([6, len(sim_vals)])
	for t in range(topn): 
		print("top %d | sim: %.6g | id: %s, file: %s" % (t+1, sim_vals[t][2], sim_vals[t][0], sim_vals[t][1]))
	tt = range(topn)
	tt.reverse()
	for t in tt:
		os.system('open %s' % sim_vals[t][1])


def car_salon_girl_demo(feats_test_file, feats_train_file):
	test_feats = image_retrieval.read_features(feats_test_file)
	train_feats = image_retrieval.read_features(feats_train_file)
	
	v_car = np.array(test_feats[0][2])
	v_girl = np.array(test_feats[1][2])
	v_car_and_girl = np.array(test_feats[2][2])
	v1 = v_car + v_girl
	v2 = v_car_and_girl - v_girl

	v_query = v_car_and_girl
#	v_query = v1
#	v_query = v2
#	v_query = v_girl
#	v_query = v_car
	
	sim_vals = image_retrieval_by_feat(v_query, test_feats, sim_type): 
	topn = np.min([6, len(sim_vals)])
	for t in range(topn): 
		print("top %d | sim: %.6g | id: %s, file: %s" % (t+1, sim_vals[t][2], sim_vals[t][0], sim_vals[t][1]))
	tt = range(topn)
	tt.reverse()
	for t in tt:
		os.system('open %s' % sim_vals[t][1])



if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("usage: cat_dog.py --cat_dog <test_feats_file> <train_feats_file>")
		print("                  --car_salon_girl <test_feats_file> <train_feats_file>")
		exit(-1)

	if sys.argv[1] == "--cat_dog":
		cat_dog_demo(sys.argv[2], sys.argv[3])
	elif sys.argv[1] == "--car_salon_girl":
		car_salon_girl_demo(sys.argv[2], sys.argv[3])


	exit(0)

