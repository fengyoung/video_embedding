# -*- coding: utf-8 -*- 
# file: image2vec.py 
# python3 supported only 
# 
# Implement of image feature vector extraction based on Inception-v3.
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


def create_graph(graph_file):
	""""Creates a graph from saved GraphDef file and returns a saver.
	
	Args:
		graph_file: A string. The graph file of inception-v3 model

	"""
	# Creates graph from saved graph_def.pb.
	with gfile.FastGFile(graph_file, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


def feat_detect_demo(graph_file, images, out_file):
	"""Implement of image feature vector extraction based on Inception-v3. 
	
	Args:
		graph_file: A string. The graph file of inception-v3 model
		images: image file list
		out_file: output file path
	"""
	cnt = 0
	fp = open(out_file, 'w')

	create_graph(graph_file)
	
	with tf.Session() as sess:
		for image in images:
			if gfile.Exists(image) == True:
				print("(%d) extracting from %s" % (cnt+1, image))
				image_data = gfile.FastGFile(image, 'rb').read()
				feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
				feats = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})
				feats = np.squeeze(feats)
				fp.write(str(cnt+1) + '\t' + image + '\t' + reduce(lambda x, y: str(x) + ',' + str(y), feats) + '\n')
			else:
				print("(%d) file does exist %s!" % (cnt+1, image))
				tf.logging.fatal('File does not exist %s', image)
				continue
			cnt += 1
		print("OK, %d images have been processed!" % cnt)
	fp.close()


if __name__ == '__main__':
	if len(sys.argv) != 5:
		print('usage: %s <model> <image_root_path> <image_suffix> <out_file>' % sys.argv[0])
		exit(-1)

	images = util.get_filepaths(sys.argv[2], sys.argv[3])
	feat_detect_demo(sys.argv[1], images, sys.argv[4])

	exit(0)


