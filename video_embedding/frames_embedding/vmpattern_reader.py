# -*- coding: utf-8 -*- 
# file: vmpattern_reader.py 
# python3 supported only 
# 
# video-matrix pattern (VMP) reader, construct pipeline and pre-read vmp from pattern/tfrecord files    
# 
# 2017-04-07 by fengyoung(fengyoung1982@sina.com)
#


import tensorflow as tf
import numpy as np
import sys
import time


def decode_vmpattern_values(values):
	"""Parse the vmp(video-matrix pattern) string proto. 

	Args:
		values: A 1-D tensor of shape [batch] and type string. The batch input of vmp strings

	vmp string proto, labels are one hot encoded:
		------------------------------------
		mid,0_0_1_0...0_0,height_width,x0_x1_x2_..._xn
		mid,0_1_1_0...0_0,height_width,x0_x1_x2_..._xn
		...
		------------------------------------

	Returns:
		Tuple of (mid_batch, label_batch, feat_batch)
		mid_batch is a 2-D tensor of shape [batch, 1] and type string 
		label_batch is a 2-D tensor of shape [batch, label_cnt] and type float32
		feat_batch is a 4-D tensor of shape [batch, height, width, channel(1)] and type float32
	"""
	batch_size = tf.size(values)
	value_seg = tf.transpose(tf.reshape(tf.string_split(values, ',').values, [batch_size, -1]))
	
	mids = tf.reshape(value_seg[0], [batch_size, -1])

	labels = tf.string_to_number(tf.string_split(value_seg[1], delimiter = '_').values, out_type = tf.float32)
	labels = tf.reshape(labels, [batch_size, -1])

	sizes = tf.string_to_number(tf.string_split(value_seg[2], delimiter = '_').values, out_type = tf.int32)	
	sizes = tf.reshape(sizes, [batch_size, -1])

	feats = tf.string_to_number(tf.string_split(value_seg[3], delimiter = '_').values, out_type = tf.float32)	
	feats = tf.reshape(feats, [batch_size, sizes[0][0], sizes[0][1], -1])

	return (mids, labels, feats)


def prepare_read_from_pattern(patt_files, batch_size = 1, max_epochs = None, shuffle = False): 
	"""Prepare to read patterns in vmp string proto from \".pattern\" file.

	Args:
		patt_files: list of pattern files with suffix \".pattern\". 
		batch_size: An `integer`. The batch size of one reading opreation. 
		max_epochs: An `integer`. max epoch num of reading 
		shuffle: Boolean. If true, the patt_files list should be randomly shuffled within each epoch. 
 
	File format:
		One vmp string proto each line	

	Returns
		Tuple of (mid_batch, label_batch, feat_batch)
		mid_batch is a 2-D tensor of shape [batch, 1] and type string 
		label_batch is a 2-D tensor of shape [batch, label_cnt] and type float32
		feat_batch is a 4-D tensor of shape [batch, height, width, channel(1)] and type float32
	"""
	# put file name string to a queue 
	filename_queue = tf.train.string_input_producer(patt_files, shuffle = shuffle, num_epochs = max_epochs)
	# create a reader from file queue
	reader = tf.TextLineReader()
	_, values = reader.read_up_to(filename_queue, batch_size)
	# decode values by self-definition
	mid_batch, label_batch, feat_batch = decode_vmpattern_values(values)
	return (mid_batch, label_batch, feat_batch)



def prepare_read_from_tfrecord(tfrecord_files, num_labels, v_height, v_width, batch_size = 1, max_epochs = None, num_threads = 4, shuffle = False): 
	"""Prepare to read patterns in tfrecord proto from \".tfrecord\" file.

	Args:
		tfrecord_files: list of pattern files with suffix \".tfrecord\". There stand tfrecord files
		num_labels: An `integer`. Number of label size 
		v_height & v_width: Two `integer`s. Denode the size of 2-D features	
		batch_size: An `integer`. The batch size of one reading opreation. 
		max_epochs: An `integer`. Max epoch num of reading 
		num_threads: An `integer`. Number of reading threads
		shuffle: Boolean. If true, the patt_files list should be randomly shuffled within each epoch. 
 
	File format:
		Features are stored as tensorflow.Example protocol buffers. The Example proto is:
		--------------------------------------------------------------------------------------
		features: {
		  feature: {
		    key: "mid"
		    value: {
		      bytes_list: {
		        value: [mid string]
		      }
		    }
		  }
		  feature: {
		    key: "label"
		    value: {
		      float_list: {
		        value: [num_labels float values of 0. or 1.] 
		      }
		    }
		  }
		  feature: {
		    key: "size"
		    value: {
		      int64_list: {
		        value: [v_height, v_width]
		      }
		    }
		  }
		  feature: {
		    key: "feature"
		    value: {
		      float_list: {
		        value: [(v_height * width) float features]
		      }
		    }
		  }
		}
		--------------------------------------------------------------------------------------

	Returns
		Tuple of (mid_batch, label_batch, feat_batch)
		mid_batch is a 2-D tensor of shape [batch, 1] and type string 
		label_batch is a 2-D tensor of shape [batch, label_cnt] and type float32
		feat_batch is a 4-D tensor of shape [batch, height, width, channel(1)] and type float32
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
			'label': tf.FixedLenFeature([num_labels], tf.float32),
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
	return (mid_batch, label_batch, feat_batch)



