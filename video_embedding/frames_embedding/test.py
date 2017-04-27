import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
import time
import os
import json
from comm import util
import vmpattern_reader as vmp_reader
import vmpattern_reader_v2 as vmp_reader_v2



def test_prepare_read_from_pattern(vmpattern_path): 
	vmpattern_files = util.get_filepaths(vmpattern_path, 'pattern')
	mids, y_0, x = vmp_reader.prepare_read_from_pattern(vmpattern_files, batch_size = 10, max_epochs = None)

	# run graph
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		
		try: 
			while not coord.should_stop():
				mid_batch, y_0_batch, x_batch, = sess.run([mids, y_0, x])
				print('*** pattern v1')
				print(mid_batch)
				print(y_0_batch)
				print('---')
				print(np.shape(mid_batch))
				print(np.shape(y_0_batch))
				print(np.shape(x_batch))
		except tf.errors.OutOfRangeError:
			print("Done!")
		finally:
			coord.request_stop()
		coord.join(threads)
		time.sleep(2)


def test_prepare_read_from_pattern_v2(vmpattern_path):
	vmpattern_files = util.get_filepaths(vmpattern_path, 'pattern')
	mids, y_0, x = vmp_reader_v2.prepare_read_from_pattern(vmpattern_files, 28, batch_size = 10, max_epochs = None) 

	# run graph
	with tf.Session() as sess: 
		
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		
		try: 
			while not coord.should_stop():
				mid_batch, y_0_batch, x_batch, = sess.run([mids, y_0, x])
				print('*** pattern v2')
				print(mid_batch)
				print(y_0_batch)
				print('---')
				print(np.shape(mid_batch))
				print(np.shape(y_0_batch))
				print(np.shape(x_batch))
		except tf.errors.OutOfRangeError:
			print("Done!")
		finally:
			coord.request_stop()
		coord.join(threads)
		time.sleep(2)


def test_prepare_read_from_tfrecord(vmpattern_path): 
	vmpattern_files = util.get_filepaths(vmpattern_path, 'tfrecord')
	mids, y_0, x = vmp_reader.prepare_read_from_tfrecord(vmpattern_files, 28, 30, 2048, batch_size = 10, max_epochs = None, shuffle = True)
	global_init = tf.global_variables_initializer()
	
	# run graph
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		
		try: 
			while not coord.should_stop():
				mid_batch, y_0_batch, x_batch, = sess.run([mids, y_0, x])
				print('*** tfrecord v1')
				print(mid_batch)
				print(y_0_batch)
				print('---')
				print(np.shape(mid_batch))
				print(np.shape(y_0_batch))
				print(np.shape(x_batch))
		except tf.errors.OutOfRangeError:
			print("Done!")
		finally:
			coord.request_stop()
		coord.join(threads)
		time.sleep(2)


def test_prepare_read_from_tfrecord_v2(vmpattern_path): 
	vmpattern_files = util.get_filepaths(vmpattern_path, 'tfrecord')
	mids, y_0, x = vmp_reader_v2.prepare_read_from_tfrecord(vmpattern_files, 28, 30, 2048, batch_size = 10, max_epochs = None, shuffle = True)
	global_init = tf.global_variables_initializer()
	
	# run graph
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		
		try: 
			while not coord.should_stop():
				mid_batch, y_0_batch, x_batch, = sess.run([mids, y_0, x])
				print('*** tfrecord v2')
				print(mid_batch)
				print(y_0_batch)
				print('---')
				print(np.shape(mid_batch))
				print(np.shape(y_0_batch))
				print(np.shape(x_batch))
		except tf.errors.OutOfRangeError:
			print("Done!")
		finally:
			coord.request_stop()
		coord.join(threads)
		time.sleep(2)



def test_auc():
	preds = tf.random_uniform((10, 3), maxval = 1, dtype = tf.float32, seed = 1)
	labels = tf.random_uniform((10, 3), maxval = 2, dtype = tf.int64, seed = 1)
	auc, update_op = tf.contrib.metrics.streaming_auc(preds, labels)

	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
	
		for _ in range(100):
			print(sess.run([preds, labels]))
			sess.run(update_op)
		print(auc.eval())


if __name__ == '__main__':
	#test_prepare_read_from_pattern("../../data/weibo_vfeature_MCN_14k_confirm30_pattern_v1/") 
	test_prepare_read_from_pattern_v2("../../data/weibo_vfeature_MCN_14k_confirm30_pattern_v2/") 
	
	#test_prepare_read_from_tfrecord("../../data/weibo_vfeature_MCN_14k_confirm30_tfrecord_v1/") 
	#test_prepare_read_from_tfrecord_v2("../../data/weibo_vfeature_MCN_14k_confirm30_tfrecord_v2_2/") 

	exit(0)


