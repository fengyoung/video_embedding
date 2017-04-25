# -*- coding: utf-8 -*- 
# file: video2vec.py 
# python3 supported only 
# 
# Implement of video level feature vector extraction based on FCNN. 
# 
# 2017-04-19 by fengyoung(fengyoung1982@sina.com)
#

import tensorflow as tf
import numpy as np
import os
import sys
import json
import time
from functools import reduce
import fcnn
import util
import vmpattern_reader as vmp_reader


def save_features(mids, feats, out_file, append = True):
	"""Saves extracted features to out file

	Args:
		mids: A 2-D tensor of shape [batch, 1] and type string. The mid of each feature vector
		feats: A 2-D tensor of shape [batch, vec_size] and type float32. Feature vectors

	Returns: 
		If successful return the number of feature vectors, otherwise return -1 
	"""
	try:
		if append:
			fp = open(out_file, 'a')
		else:
			fp = open(out_file, 'w')
		batch_size = mids.shape[0]	
		for i in range(batch_size): 
			fp.write(mids[i][0].decode() + '\t' + reduce(lambda x, y: str(x) + ',' + str(y), feats[i]) + '\n')	
		return batch_size
	except IOError as err: 
		print("File Error: " + str(err))
		return -1
	finally:
		fp.close()


def feat_detect_demo(model_path, vmpattern_path, out_file, batch_size = 1, vmpatt_file_suffix = 'tfrecord'):
	"""Implement of video level feature vector extraction based on FCNN. 

	Args:
		model_path: A `string`. Path of model.
		vmpattern_path: A `string`. Path of vmp files.
		out_file: A `string`. Path and file name of output file. 
		batch_size: size of batch extraction.
		vmpatt_file_suffix: A `string`. Suffix of vmp files, 'tfrecord' for tfrecord prot or 'pattern' for vmp string proto

	Out file format:
		--------------------------------------------------------------------------------------
		mid0\tf0,f1,...,fn
		mid1\tf0,f1,...,fn
		mid2\tf0,f1,...,fn
		...
		--------------------------------------------------------------------------------------
	"""
	# constructs the FCNN model and create by reading arch_file	
	fcnn_model = fcnn.FCNN()
	if not fcnn_model.read_arch(os.path.join(model_path, fcnn.g_arch_file_name)):
		print("Error: failed to load FCNN architecture")
		return
	
	# prepare to read video-matrix patterns
	if vmpatt_file_suffix == 'tfrecord':
		vmpattern_files = util.get_filepaths(vmpattern_path, vmpatt_file_suffix) 
		if len(vmpattern_files) == 0:
			print("Error: path \"%s\" is empty or not exist!" % vmpattern_path) 
			return
		mid_batch, y_0, x = vmp_reader.prepare_read_from_tfrecord(vmpattern_files, fcnn_model.arch["out_size"], fcnn_model.arch["in_height"], fcnn_model.arch["in_width"],   
			batch_size = batch_size, max_epochs = 1, shuffle = False)
	elif vmpatt_file_suffix == 'pattern':
		vmpattern_files = util.get_filepaths(vmpattern_path, vmpatt_file_suffix) 
		if len(vmpattern_files) == 0:
			print("Error: path \"%s\" is empty or not exist!" % vmpattern_path) 
			return
		mid_batch, y_0, x = vmp_reader.prepare_read_from_pattern(vmpattern_files, batch_size = batch_size, max_epochs = 1, shuffle = False)
	else: 
		print("Error: unrecognized suffix \"%s\"" % vmpatt_file_suffix)
		return

	# create saver for FCNN restore
	saver = tf.train.Saver()

	# draw training graph
	f = fcnn_model.feature_detect(x)
	append = False
	ready_cnt = 0

	# run graph
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		
		# restore the model
		saver.restore(sess = sess, save_path = tf.train.latest_checkpoint(model_path))

		try:
			while not coord.should_stop():
				start_time = time.time()
				mids, feats = sess.run([mid_batch, f])
				cnt = save_features(mids, feats, out_file, append = append)
				if cnt < 0:
					print("Error: saving error!")
					break
				append = True
				ready_cnt += cnt
				print("extracts: %d | time_cost(s): %.3f" % (ready_cnt, time.time() - start_time))
		except tf.errors.OutOfRangeError:
			print("Done!")
		finally:
			coord.request_stop()

		coord.join(threads)
		time.sleep(1)


if __name__ == '__main__':
	if len(sys.argv) != 5: 
		print("usage: video2vec.py <model_path> <vmpattern_path> <suffix> <out_file>")
		exit(-1)

	feat_detect_demo(sys.argv[1], sys.argv[2], sys.argv[4], batch_size = 50, vmpatt_file_suffix = sys.argv[3])

	exit(0)


