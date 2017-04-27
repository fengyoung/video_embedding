# -*- coding: utf-8 -*- 
# file: fcnn_pred.py 
# python3 supported only 
# 
# Implement of FCNN prediction.
# 
# 2017-04-19 by fengyoung(fengyoung1982@sina.com)
#

import sys
sys.path.append("../")

import os
import json
import time
import tensorflow as tf
import numpy as np
from comm import util
import vmpattern_reader_v2 as vmp_reader
import fcnn


def test_demo(model_path, vmpattern_path, vmpatt_file_suffix = 'tfrecord'):
	"""Implement of FCNN prediction for testing.

	Args:
		model_path: A `string`. Path of model.
		vmpattern_path: A `string`. Path of vmp files.
		vmpatt_file_suffix: A `string`. Suffix of vmp files, 'tfrecord' for tfrecord prot or 'pattern' for vmp string proto
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
		mids, y_0, x = vmp_reader.prepare_read_from_tfrecord(vmpattern_files, fcnn_model.arch["out_size"], fcnn_model.arch["in_height"], fcnn_model.arch["in_width"],   
			batch_size = 1, max_epochs = 1, shuffle = False)
	elif vmpatt_file_suffix == 'pattern':
		vmpattern_files = util.get_filepaths(vmpattern_path, vmpatt_file_suffix) 
		if len(vmpattern_files) == 0:
			print("Error: path \"%s\" is empty or not exist!" % vmpattern_path) 
			return
		mids, y_0, x = vmp_reader.prepare_read_from_pattern(vmpattern_files, fcnn_model.arch["out_size"], batch_size = 1, max_epochs = 1, shuffle = False)
	else: 
		print("Error: unrecognized suffix \"%s\"" % vmpatt_file_suffix)
		return

	# create saver for FCNN restore
	saver = tf.train.Saver()

	# draw training graph
	y = fcnn_model.classify(x)
	correct = 0
	i = 0

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
				mid, pred, label, label_pred = sess.run([mids, y, tf.argmax(y_0, 1), tf.argmax(y, 1)])
				i += 1
				if label_pred[0] == label[0]:
					print("(%d) | mid: %s, [%d] -> [%d] | pred: %.6g | time_cost(s): %.3f" % 
							(i, mid[0][0].decode(), label[0], label_pred[0], pred[0][label_pred[0]], time.time() - start_time))
					util.LOG(str.format("(%d) | mid: %s, [%d] -> [%d] | pred: %.6g | time_cost(s): %.3f" % 
								(i, mid[0][0].decode(), label[0], label_pred[0], pred[0][label_pred[0]], time.time() - start_time)), ltype = "INFO")
					correct += 1	
				else: 
					print("** (%d) | mid: %s, [%d] -> [%d] | pred: %.6g | time_cost(s): %.3f" % 
							(i, mid[0][0].decode(), label[0], label_pred[0], pred[0][label_pred[0]], time.time() - start_time))
					util.LOG(str.format("** (%d) | mid: %s, [%d] -> [%d] | pred: %.6g | time_cost(s): %.3f" % 
								(i, mid[0][0].decode(), label[0], label_pred[0], pred[0][label_pred[0]], time.time() - start_time)), ltype = "INFO") 

		except tf.errors.OutOfRangeError:
			print("Testing Done!")
		finally:
			coord.request_stop()
			print("Pr: %.6g" % (float(correct) / float(i)))

		coord.join(threads)
		time.sleep(1)



if __name__ == '__main__':
	if len(sys.argv) != 4 and len(sys.argv) != 5: 
		print("usage: fcnn_pred.py <model_path> <vmpattern_path> <suffix> <|log_file>")
		exit(-1)

	if len(sys.argv) == 5:
		util.g_log_file = sys.argv[4]

	test_demo(sys.argv[1], sys.argv[2], sys.argv[3])

	exit(0)


