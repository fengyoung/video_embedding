# -*- coding: utf-8 -*- 
# file: vmp_file.py 
# python3 supported only
# 
# Operations of Video-Mat Pattern(VMP) file. 
#
# There are 2 types of VMP file: pattern-string proto & tfrecord proto
#
# Pattern-string proto:
# -----------------------------------------------------------
# mid,labelid0_labelid1_labelid2,height_width,x0_x1_x2_..._xn
# mid,labelid2_labelid5,height_width,x0_x1_x2_..._xn
# ...
# -----------------------------------------------------------
#
# Tfrecord proto: 
# -----------------------------------------------------------
# features: {
#   feature: {
#     key: "mid"
#     value: {
#       bytes_list: {
#         value: [mid string]
#       }
#     }
#   }
#   feature: {
#     key: "label"
#     value: {
#       bytes_list: {
#         value: ["0,3,7"]
#       }
#     }
#   }
#   feature: {
#     key: "size"
#     value: {
#       int64_list: {
#         value: [v_height, v_width]
#       }
#     }
#   }
#   feature: {
#     key: "feature"
#     value: {
#       float_list: {
#         value: [(v_height * v_width) float features]
#       }
#     }
#   }
# }
# -----------------------------------------------------------
# 
# 
# 2017-06-22 by fengyoung(fengyoung1982@sina.com)
# 

import sys
sys.path.append("../")

import numpy as np
import tensorflow as tf
from embedding import util


def write_as_tfrecord(mids, labels, video_matrices, out_file, sess):
	"""Writes video matrices to file in tfrecord format.

	Args:
		mids: A list of mid in string
		labels: A list of label (integer list) which is sparse.
		video_matrics: A 3-D array of float in shape [batch_size, frames, 2048]. List of video-matrices  
		out_file: A string. output file path
		sess: tf.Session

	Returns:
		True for success, False for failed.
	"""
	batch_size = video_matrices.shape[0]
	height = video_matrices.shape[1]
	width = video_matrices.shape[2]
	if len(mids) != batch_size and len(labels) != batch_size:
		return False

	writer = tf.python_io.TFRecordWriter(out_file)
	for i in range(batch_size): 
		example = tf.train.Example(features = tf.train.Features(feature={
					"mid": tf.train.Feature(bytes_list = tf.train.BytesList(value = [mids[i].encode()])),
					"label": tf.train.Feature(bytes_list = tf.train.BytesList(value = [util.number_list_to_string(labels[i], '_').encode()])), 
					"size": tf.train.Feature(int64_list = tf.train.Int64List(value = [height, width])), 
					"feature": tf.train.Feature(float_list = tf.train.FloatList(value = np.reshape(video_matrices[i], [-1])))
					}))
		writer.write(example.SerializeToString())
	
	return True	


def write_as_pattern_string(mids, labels, video_matrices, out_file, append = False): 
	"""Writes video matrices to file in pattern-string format.

	Args:
		mids: A list of mid
		labels: A list of label (integer list) which is sparse.
		video_matrics: A 3-D array of float in shape [batch_size, frames, 2048]. List of video-matrices  
		out_file: A string. Output file path
		append: A boolean. If appending

	Returns:
		True for success, False for failed.
	"""
	batch_size = video_matrices.shape[0]
	height = video_matrices.shape[1]
	width = video_matrices.shape[2]
	if len(mids) != batch_size or len(labels) != batch_size :
		return False

	if append: 
		fp = open(out_file, 'a')
	else: 
		fp = open(out_file, 'w')

	try:
		for i in range(batch_size):
			ss_info = "%s,%s,%d_%d," % (mids[i], util.number_list_to_string(labels[i], '_'), height, width)
			ss_feat = util.number_list_to_string(list(map(lambda x: round(x, 6), np.reshape(video_matrices[i], [-1]))), '_')
			fp.write(ss_info + ss_feat + '\n')
		return True
	except IOError as err:
		return False
	finally:
		fp.close()


def video_vec_write_as_pattern_string(mids, labels, video_vectors, out_file, append = False): 
	"""Writes video vectors to file in pattern-string format.
		The video vector should be transformed as a video matrix in shape [1, n]. 
	
	Args:
		mids: A list of mid
		labels: A list of label (integer list) which is sparse.
		video_vectors: A 2-D array of float in shape [batch_size, n]. List of video-vectors
		out_file: A string. Output file path
		append: A boolean. If appending

	Returns:
		True for success, False for failed.
	"""
	return write_as_pattern_string(mids, labels, np.reshape(video_vectors, [video_vectors.shape[0], 1, video_vectors.shape[1]]), out_file, append)


def label_string_to_dense_one_hots(lablel_strings, num_labels):
	"""Converts label string to densely one hots tensor

	Args: 
		labels_strings: A 1-D tensor of shape [batch_size] and type string. 
		num_labels: An `integer`. Number of label size 
	
	Returns:
		2-D tensor of shape [batch_size, num_labels] and type float32.
	"""
	st = tf.string_split(lablel_strings, '_')
	label_sparse = tf.SparseTensor(indices = st.indices, values = tf.string_to_number(st.values, out_type = tf.int32), dense_shape = st.dense_shape)
	return tf.reduce_sum(tf.one_hot(tf.sparse_tensor_to_dense(label_sparse, default_value = -1), depth = num_labels, on_value = 1.0, off_value = 0.0, axis = -1), axis = 1)


def prepare_read_from_tfrecord(tfrecord_files, num_labels, v_height, v_width, batch_size = 1, max_epochs = None, num_threads = 4, shuffle = False): 
	"""Prepares to read vmp in tfrecord proto files.

	Args:
		tfrecord_files: list of vmp files in tfrecord proto. 
		num_labels: An `integer`. Number of label size 
		v_height & v_width: Two `integer`s. Denode the size of 2-D features	
		batch_size: An `integer`. The batch size of one reading opreation. 
		max_epochs: An `integer`. Max epoch num of reading 
		num_threads: An `integer`. Number of reading threads
		shuffle: Boolean. If true, the patt_files list should be randomly shuffled within each epoch. 
	
	Returns:
		Tuple of (mid_batch, label_batch, feat_batch)
		mid_batch is a 2-D tensor of shape [batch_size, 1] and type string 
		label_batch is a 2-D tensor of shape [batch_size, num_labels] and type float32. The label of each video is one hot encoded densely. 
		feat_batch is a 4-D tensor of shape [batch_size, height, width, channel(1)] and type float32
	"""
	# put file name string to a queue 
	filename_queue = tf.train.string_input_producer(tfrecord_files, shuffle = shuffle, num_epochs = max_epochs)
	#  create a reader from file queue
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue) 
	# parse the record
	features = tf.parse_single_example(serialized_example, 
			features = {
			'mid': tf.FixedLenFeature([1], tf.string),
			'label': tf.FixedLenFeature([1], tf.string),
			'feature': tf.FixedLenFeature([v_height * v_width], tf.float32)})
	mid_out = features['mid']
	label_out = features['label']
	feat_out = tf.reshape(features['feature'], [v_height, v_width, 1])
	# data padding via batch
	if shuffle:
		mid_batch, label_batch, feat_batch = tf.train.shuffle_batch([mid_out, label_out, feat_out], batch_size = batch_size, 
									capacity = batch_size * 500, min_after_dequeue = batch_size * 100, num_threads = num_threads)
	else: 
		mid_batch, label_batch, feat_batch = tf.train.batch([mid_out, label_out, feat_out], batch_size = batch_size, 
									capacity = batch_size * 500, num_threads = num_threads)
	label_batch = label_string_to_dense_one_hots(tf.reshape(label_batch, [-1]), num_labels)
	return (mid_batch, label_batch, feat_batch)


def decode_vmp_pattern_string(values, num_labels):
	"""Parses the vmp(video-matrix pattern) pattern-string proto. 

	Args:
		values: A 1-D tensor of shape [batch_size] and type string. The batch input of vmp pattern-strings
		num_labels: An integer. Number of label size 

	Returns:
		Tuple of (mid_batch, label_batch, feat_batch)
		mid_batch is a 2-D tensor of shape [batch_size, 1] and type string 
		label_batch is a 2-D tensor of shape [batch_size, num_labels] and type float32. The label of each video is one hot encoded. 
		feat_batch is a 4-D tensor of shape [batch_size, height, width, channel(1)] and type float32
	"""
	batch_size = tf.size(values)
	value_seg = tf.transpose(tf.reshape(tf.string_split(values, ',').values, [batch_size, -1]))
	
	mid_batch = tf.reshape(value_seg[0], [batch_size, -1])

	llabel_batch = label_string_to_dense_one_hots(value_seg[1], num_labels)

	sizes = tf.string_to_number(tf.string_split(value_seg[2], delimiter = '_').values, out_type = tf.int32)	
	sizes = tf.reshape(sizes, [batch_size, -1])

	feat_batch = tf.string_to_number(tf.string_split(value_seg[3], delimiter = '_').values, out_type = tf.float32)	
	feat_batch = tf.reshape(feat_batch, [batch_size, sizes[0][0], sizes[0][1], -1])

	return (mid_batch, label_batch, feat_batch)


def parse_pattern_string(value): 
	"""Parses a string in pattern-string proto.

	Args: 
		value: A string. String in pattern-string proto
	
	Return: 
		Tuple of (mid, label, feat)
		feat is 2-D array of shape [v_height, v_width] 	
	"""
	ss = value.split(',')	
	mid = ss[0]
	label = util.string_to_integer_list(ss[1], delimiter = '_')
	size = util.string_to_integer_list(ss[2], delimiter = '_')
	feat = util.string_to_float_list(ss[3], delimiter = '_')
	feat = np.reshape(feat, size)
	return (mid, label, feat)


def prepare_read_from_pattern_string(patt_files, num_labels, batch_size = 1, max_epochs = None, shuffle = False): 
	"""Prepare to read patterns in vmp in pattern-string proto.

	Args:
		patt_files: list of vmp files in pattern-string proto. 
		num_labels: An `integer`. Number of label size 
		batch_size: An `integer`. The batch size of one reading opreation. 
		max_epochs: An `integer`. max epoch num of reading 
		shuffle: Boolean. If true, the patt_files list should be randomly shuffled within each epoch. 
 
	Returns
		Tuple of (mid_batch, label_batch, feat_batch)
		mid_batch is a 2-D tensor of shape [batch_size, 1] and type string 
		label_batch is a 2-D tensor of shape [batch_size, label_cnt] and type float32
		feat_batch is a 4-D tensor of shape [batch_size, height, width, channel(1)] and type float32
	"""
	# put file name string to a queue 
	filename_queue = tf.train.string_input_producer(patt_files, shuffle = shuffle, num_epochs = max_epochs)
	# create a reader from file queue
	reader = tf.TextLineReader()
	_, values = reader.read_up_to(filename_queue, batch_size)
	# decode values by self-definition
	mid_batch, label_batch, feat_batch = decode_vmp_pattern_string(values, num_labels)
	return (mid_batch, label_batch, feat_batch)


