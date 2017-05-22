# -*- coding: utf-8 -*- 
# file: image_retrieval.py 
# python3 supported only 
# 
# Image Retrieval demo based on Image-Embedding. 
# 
# 2017-04-19 by fengyoung(fengyoung1982@sina.com)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

import os.path
import re
import tarfile

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile
from comm import util
from comm import similary
import image2vec 


def read_features(image_feat_file):
	"""Load all image feature vectors

	Args: 
		image_feat_file: A string. Image features file path

	Returns:
		The list of image feature vectors [[id, image_path, [vector]], ...]
	"""
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


def image_sim_match(feat1, feat2, sim_type = 'cosine'):
	"""Caculate similary between two image feature vectors

	Args: 
		feat1/feat2: List of float. Dense Image feature vector.
		sim_type: A string. Type of the similary method. Default is 'cosine'

	Returns:
		The similary value
	"""
	if sim_type == 'cosine':
		return similary.cosine_sim(feat1, feat2)
	elif sim_type == 'euclidean':
		return similary.euclidean_sim(feat1, feat2)
	elif sim_type == 'manhattan':
		return similary.manhattan_sim(feat1, feat2)
	elif sim_type == 'chebyshev':
		return similary.chebyshev_sim(feat1, feat2)
	else:
		return False

		
def run_image_sim_match(graph_file, image, image_feats): 
	image2vec.create_graph(graph_file)
	with tf.Session() as sess:
		if gfile.Exists(image) == True:
			image_data = gfile.FastGFile(image, 'rb').read()
			feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
			feats = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})
			feats = np.squeeze(feats)
			for sim_type in ['cosine', 'euclidean', 'manhattan', 'chebyshev']:
				sim_vals = []	
				for im in image_feats:
					sim = image_sim_match(feats, im[2], sim_type)
					sim_vals.append([im[0], im[1], sim])
				sim_vals = sorted(sim_vals, cmp = lambda x, y: cmp(x[2], y[2]), reverse = True)
				print("SIM_TYPE: %s" % sim_type)
				print("IMAGE: %s" % image)
				for t in range(np.min([5, len(sim_vals)])):
					print("top %d | sim: %.6g | id: %s, file: %s" % (t+1, sim_vals[t][2], sim_vals[t][0], sim_vals[t][1]))
		else:
			print("file does exist %s!" % image)
			tf.logging.fatal('File does not exist %s', image)


def run_image_sim_match2(graph_file, image, image_feats, sim_type): 
	image2vec.create_graph(graph_file)
	with tf.Session() as sess:
		if gfile.Exists(image) == True:
			image_data = gfile.FastGFile(image, 'rb').read()
			feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
			feats = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})
			feats = np.squeeze(feats)
			sim_vals = []	
			for im in image_feats:
				sim = image_sim_match(feats, im[2], sim_type)
				sim_vals.append([im[0], im[1], sim])
			sim_vals = sorted(sim_vals, cmp = lambda x, y: cmp(x[2], y[2]), reverse = True)
			print("SIM_TYPE: %s" % sim_type)
			print("IMAGE: %s" % image)
			topn = np.min([5, len(sim_vals)])
			for t in range(topn): 
				print("top %d | sim: %.6g | id: %s, file: %s" % (t+1, sim_vals[t][2], sim_vals[t][0], sim_vals[t][1]))
			tt = range(topn)
			tt.reverse()
			for t in tt:
				os.system('open %s' % sim_vals[t][1])
			os.system('open %s' % image)
		else:
			print("file does exist %s!" % image)
			tf.logging.fatal('File does not exist %s', image)


if __name__ == '__main__':
	if len(sys.argv) != 4 and len(sys.argv) != 5:
		print('usage: %s <model> <image_features> <image> <|sim_type>' % sys.argv[0])
		print('sim_type: cosine, euclidean, manhattan or chebyshev')
		exit(-1)

	image_feats = read_features(sys.argv[2])

	if len(sys.argv) == 4:
		run_image_sim_match(sys.argv[1], sys.argv[3], image_feats)
	elif len(sys.argv) == 5:
		run_image_sim_match2(sys.argv[1], sys.argv[3], image_feats, sys.argv[4])

	exit(0)


