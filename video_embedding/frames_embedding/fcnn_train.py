# -*- coding: utf-8 -*- 
# file: fcnn_train.py 
# python3 supported only 
# 
# Implement of FCNN training. 
# 
# 2017-04-19 by fengyoung(fengyoung1982@sina.com)
#

import sys
sys.path.append("../")

import os
import json
import time
import fcnn
import tensorflow as tf
import numpy as np
from comm import util
import vmpattern_reader_v2 as vmp_reader


def read_config(config_file):
	"""Read configure from file, and parse config parameters from json format

	Args:
		config_file: A `string`. Path and name of configure file

	Returns:
		Tuple of 2 configure dicts, one for FCNN architecture and the other for training
	"""
	try: 
		fp = open(config_file, 'r')
		json_str = ""	
		for line in fp:
			if len(line.lstrip().rstrip()) > 0: 
				json_str += line.lstrip().rstrip()
		conf = json.loads(json_str)
		return (conf["fcnn_arch"], conf["train_params"])
	except IOError as err:
		print("File Error: " + str(err))
		return None
	finally:
		fp.close()


def train_demo(config_file, vmpattern_path, out_model_path, vmpatt_file_suffix = 'tfrecord'): 
	"""Implement of FCNN training.

	Args:
		config_file: A `string`. Path and name of configure file.
		vmpattern_path: A `string`. Path of vmp files.
		out_model_path: A `string`. Path of model output.
		vmpatt_file_suffix: A `string`. Suffix of vmp files, 'tfrecord' for tfrecord prot or 'pattern' for vmp string proto
	"""
	if os.path.exists(out_model_path):
		print("Error: dir \"%s\" is exist!" % out_model_path)
		util.LOG(str.format("dir \"%s\" is exist" % out_model_path), ltype = "ERROR")
		return
	os.makedirs(out_model_path)

	# get vmpattern_files queue
	vmpattern_files = util.get_filepaths(vmpattern_path, vmpatt_file_suffix) 
	if len(vmpattern_files) == 0:
		print("Error: path \"%s\" is empty or not exist!" % vmpattern_path) 
		util.LOG(str.format("path \"%s\" is empty or not exist" % vmpattern_path), ltype = "ERROR")
		return
	
	# read config file
	ret = read_config(config_file)
	if not ret: 
		print("Error: failed to read config from \"%s\"" % config_file)
		util.LOG("failed to read config from \"" + config_file + "\"", ltype = "ERROR")
		return
	arch_params, train_params = ret
	
	# prepare to read video-matrix patterns
	if vmpatt_file_suffix == 'tfrecord':
		mids, y_0, x = vmp_reader.prepare_read_from_tfrecord(vmpattern_files, arch_params["out_size"], arch_params["in_height"], arch_params["in_width"],   
			batch_size = train_params["batch_size"], max_epochs = train_params["max_epochs"], shuffle = train_params["shuffle"])
	elif vmpatt_file_suffix == 'pattern':
		mids, y_0, x = vmp_reader.prepare_read_from_pattern(vmpattern_files, arch_params["out_size"],  
			batch_size = train_params["batch_size"], max_epochs = train_params["max_epochs"], shuffle = train_params["shuffle"])
	else: 
		print("Error: unrecognized suffix \"%s\"" % vmpatt_file_suffix)
		util.LOG(str.format("unrecognized suffix \"%s\"" % vmpatt_file_suffix), ltype = "ERROR")
		return
 
	# create the FCNN model	
	fcnn_model = fcnn.FCNN(arch_params)

	# draw training graph
	y = fcnn_model.propagate_to_classifier(x)
	# cross entropy as loss	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_0, logits = y))
	# Adam optimization 
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	# trainning accuracy 	
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_0, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# create saver			
	saver = tf.train.Saver()

	# run graph
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)

		i = 0
		max_accuracy = 0.0
		min_loss = 999999999.9
		ss1 = 0
		ss2 = 0
		time_cost = 0.0
		batch_iter_time_cost = 0.0
		NUM_BATCH_ITER = 64 
		
		try: 
			print("Training Start!")
			util.LOG("Training Start!", ltype = "INFO")
			while not coord.should_stop():
				iter_start_time = time.time() 
				train_step.run()
				i += 1

				if i % NUM_BATCH_ITER == 0:
					loss, train_accuracy = sess.run([cross_entropy, accuracy])
					batch_iter_time_cost += (time.time() - iter_start_time)
				
					ss1 += 1
					ss2 += 1
					if loss < min_loss: 
						min_loss = loss
						ss1 = 0
					if train_accuracy  > max_accuracy: 
						max_accuracy = train_accuracy
						ss2 = 0

					print("batch_iter %d | num_iter %d | loss: %.6g, ss: %d | accuracy %.6g, ss: %d | time_cost(s): %.3f" % 
						(i / NUM_BATCH_ITER, i, loss, ss1, train_accuracy, ss2, batch_iter_time_cost))
					util.LOG(str.format("batch_iter %d | num_iter %d | loss: %.6g, ss: %d | accuracy %.6g, ss: %d | time_cost(s): %.3f" % 
						(i / NUM_BATCH_ITER, i, loss, ss1, train_accuracy, ss2, batch_iter_time_cost)), ltype = "INFO")

					time_cost += batch_iter_time_cost 
					batch_iter_time_cost = 0.0
			
					if train_params["early_stop"] > 0 and ss1 >= train_params["early_stop"]: 
						print("Early Stop, Training Done!")
						util.LOG("Early Stop, Training Done!", ltype = "INFO")
						break
					if loss < train_params["epsilon"]:
						print("Small Loss, Training Done!")
						util.LOG("Small Loss, Training Done!", ltype = "INFO")
						break
				else:
					batch_iter_time_cost += (time.time() - iter_start_time)
				
				if i % 100 == 0:
					saver.save(sess = sess, save_path = os.path.join(out_model_path, fcnn.g_model_file_name), global_step = i)
					fcnn_model.save_arch(os.path.join(out_model_path, fcnn.g_arch_file_name))
		except tf.errors.OutOfRangeError:
			print("Epochs Endding, Training Done!")
			util.LOG("Epochs Endding, Training Done!", ltype = "INFO")
			time_cost += batch_iter_time_cost
		finally:
			out_path = saver.save(sess = sess, save_path = os.path.join(out_model_path, fcnn.g_model_file_name), global_step = i)
			fcnn_model.save_arch(os.path.join(out_model_path, fcnn.g_arch_file_name))
			coord.request_stop()
			
			hour = int(int(time_cost) / 3600)
			minute = int(int(time_cost - hour * 3600.) / 60)
			sec = int(time_cost - hour * 3600. - minute * 60.)
			
			print("Model path is %s" % out_path) 
			util.LOG(str.format("Model path is %s" % out_path), ltype = "INFO") 
			print("Total time cost: %d:%d:%d" % (hour, minute, sec))
			util.LOG(str.format("Total time cost: %02d:%02d:%02d" % (hour, minute, sec)), ltype = "INFO") 
		coord.join(threads)
		time.sleep(2)


if __name__ == '__main__':
	if len(sys.argv) != 4 and len(sys.argv) != 5: 
		print("usage: fcnn_train.py <config_file> <vmpattern_path(!!tfrecord only)> <out_model_path> <|log_file>")
		exit(-1)
	
	if len(sys.argv) == 5:
		util.g_log_file = sys.argv[4]

	train_demo(sys.argv[1], sys.argv[2], sys.argv[3])
	exit(0)



