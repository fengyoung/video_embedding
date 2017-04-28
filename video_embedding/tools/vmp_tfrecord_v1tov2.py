# -*- coding: utf-8 -*- 
# file: vmp_tfrecord_v1tov2.py 

import sys
sys.path.append("../")

import os
import functools
import time
import numpy as np
import tensorflow as tf
from comm import util 



def prepare_read_from_tfrecord(tfrecord_files, num_labels, v_height, v_width, batch_size = 1, max_epochs = None, num_threads = 4, shuffle = False): 
	# put file name string to a queue 
	filename_queue = tf.train.string_input_producer(tfrecord_files, shuffle = shuffle, num_epochs = max_epochs)
	#  create a reader from file queue
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue) 
	# parse the record
	features = tf.parse_single_example(serialized_example, 
			features = {
			'mid': tf.FixedLenFeature([1], tf.string),
			'off': tf.FixedLenFeature([1], tf.int64),
			'label': tf.FixedLenFeature([num_labels], tf.float32),
			'size': tf.FixedLenFeature([2], tf.int64),
			'feature': tf.FixedLenFeature([v_height * v_width], tf.float32)})
	mid_out = features['mid']
	off_out = features['off']
	label_out = features['label']
	size_out = features['size']
	feat_out = tf.reshape(features['feature'], [v_height, v_width, 1])
	# data padding via batch
	if shuffle:
		mid_batch, off_batch, label_batch, size_batch, feat_batch = tf.train.shuffle_batch([mid_out, off_out, label_out, size_out, feat_out], batch_size = batch_size, 
									capacity = batch_size * 500, min_after_dequeue = batch_size * 100, num_threads = num_threads)
	else: 
		mid_batch, off_batch, label_batch, size_batch, feat_batch = tf.train.batch([mid_out, off_out, label_out, size_out, feat_out], batch_size = batch_size, 
									capacity = batch_size * 500, num_threads = num_threads)
	return (mid_batch, off_batch, label_batch, size_batch, feat_batch)


def get_index(s_list, target_val = 1):
	ret_list = []
	for i in range(len(s_list)):
		if s_list[i] == target_val:
			ret_list.append(i)
	return ret_list


def convert_from_v1_to_v2(in_v1_path, out_v2_path, out_size, in_height, in_width): 
	if os.path.exists(out_v2_path):
		print("Error: dir \"%s\" is exist!" % out_v2_path)
		return False
	os.makedirs(out_v2_path)

	vmpattern_files = util.get_filepaths(in_v1_path, 'tfrecord')
	batch_size = 128
	mid_batch, off_batch, y_batch, size_batch, x_batch = prepare_read_from_tfrecord(vmpattern_files, out_size, in_height, in_width, batch_size = batch_size, max_epochs = 1, shuffle = False)
	cnt = 0


	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator() 	
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		try:
			while not coord.should_stop():
				out_file = os.path.join(out_v2_path, util.random_vmp_file_name(prefix = 'vmp', suffix = 'tfrecord'))
				writer = tf.python_io.TFRecordWriter(out_file)

				mids, offs, ys, sizes, xs = sess.run([mid_batch, off_batch, y_batch, size_batch, x_batch])
				
				for i in range(len(mids)):
					example = tf.train.Example(features = tf.train.Features(feature={
							"mid": tf.train.Feature(bytes_list = tf.train.BytesList(value = np.reshape(mids[i], [-1]))),
							"off": tf.train.Feature(int64_list = tf.train.Int64List(value = np.reshape(offs[i], [-1]))),
							"label": tf.train.Feature(bytes_list = tf.train.BytesList(value = [str.encode(util.number_list_to_string(get_index(np.reshape(ys[i], [-1]), 1), '_'))])),
							"size": tf.train.Feature(int64_list = tf.train.Int64List(value = np.reshape(sizes[i], [-1]))), 
							"feature": tf.train.Feature(float_list = tf.train.FloatList(value = np.reshape(xs[i], [-1])))
							}))
					writer.write(example.SerializeToString())
					cnt += 1
			
				print("(%d) current target \"%s\"" % (cnt, out_file))
		except tf.errors.OutOfRangeError:
			print("Done!")
		finally:
			coord.request_stop()

		coord.join(threads)
		time.sleep(1)



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: %s <vmp_tfrecord_path_v1> <out_path>" % sys.argv[0])
		exit(-1)

	convert_from_v1_to_v2(sys.argv[1], sys.argv[2], 28, 30, 2048)

	exit(0)



