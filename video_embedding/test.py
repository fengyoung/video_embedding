import tensorflow as tf
import numpy as np
import time
import os
import sys
import json
import util
import vmpattern_reader as vmp_reader



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
				print('--')
				print(sess.run(mids))
				print(sess.run(tf.shape(mids)))
				print(sess.run(tf.shape(y_0)))
				print(sess.run(tf.shape(x)))
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
				print('**')
				print(sess.run(mids))
				print(sess.run(tf.shape(mids)))
				print(sess.run(tf.shape(y_0)))
				print(sess.run(tf.shape(x)))
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
	#test_prepare_read_from_pattern("../data/weibo_vfeature_MCN_14k_confirm30_pattern/") 
	test_prepare_read_from_tfrecord("../data/weibo_vfeature_MCN_14k_confirm30_tfrecord/") 
	#test_auc()
	exit(0)


