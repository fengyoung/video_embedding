# -*- coding: utf-8 -*- 
# file: image_embedding.py
# python3 supported only 
#
# Image embedding based on Inception-v3
#
# An Image should be embedded to a 1-D vector in shape 2048. The vector can be used representing the content of the image.
#
# A video should be embedded to a 2-D matrix in shape [N, 2048]. N is the number of frames extracted from the video. 
# Each row of the video-matrix is the image-vector of one frame, and the whole video-matrix can be used representing the content of the video. 
#
#
# 2017-06-22 by fengyoung(fengyoung1982@sina.com)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

import os.path
import tarfile
from embedding import video_decode

# pylint: disable=unused-import,g-bad-import-order
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile


def create_graph(graph_file):
	""""Creates a graph from saved GraphDef file and returns a saver.
	
	Args:
		graph_file: A string. The graph file of inception-v3 model.
	"""
	# Creates graph from saved graph_def.pb.
	with gfile.FastGFile(graph_file, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


def feature_from_single_image_file(image_file, sess):
	""""Extracts feature vector from a single image file based on Inception-v3.
		The feature vector can represent the image content.
		Before this function, create_graph() should be called first to create graph of Inception-v3.
	
	Args:
		image_file: A string. The image file whose feature should be extracted.
		sess: tf.Session

	Returns:
		A 1-D array of float in shape [2048,]. If failed, return None.
	"""
	feature = None
	if gfile.Exists(image_file) == True:
		image_data = gfile.FastGFile(image_file, 'rb').read()
		feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		feat = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})
		feature = np.squeeze(feat)
	else:
		tf.logging.fatal('File does not exist %s', image_file)
	return feature

	
def feature_from_single_video_file(video_file, sess, start_off = 0, sampling_fps = 1, max_frame_cnt = 60, padding = True): 
	""""Extracts feature vectors from single video file based on Inception-v3.
		These vectors constitute the video-matrix in time sequence to represent the video content. 
		Before this function, create_graph() should be called first to create graph of Inception-v3.

	Args: 
		video_file: A String. Video file path.
		sess: tf.Session
		start_off: A Integer`. Start offset in second. 
		sampling_fps: A Integer. Frames per second in sampling. 
		max_frame_cnt: A Integer. How many frames should be extracted.
		padding: A Boolean. If 0 should be padded when the video was too short.

	Returns:
		A 2-D array of float in shape [frames, 2048]. If failed, return None.
	"""
	if gfile.Exists(video_file) == False:
		tf.logging.fatal('File does not exist %s', video_file)
		return None
	
	# decodes & extracts frames from the video 
	frames = video_decode.frames_extract(video_file, start_off, sampling_fps, max_frame_cnt)
	shape = frames.shape
	features = []

	# extracts feature of each frame 
	for i in range(shape[0]):
		image_data = sess.run(tf.image.encode_jpeg(tf.constant(frames[i])))
		feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		feat = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})
		features.append(np.squeeze(feat))

	# padding
	if padding and max_frame_cnt > len(features): 
		zero_feat = np.zeros([2048], dtype = np.float)	
		for _ in range(max_frame_cnt - len(features)): 
			features.append(zero_feat)	

	return np.array(features)


