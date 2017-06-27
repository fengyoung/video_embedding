# -*- coding: utf-8 -*- 
# file: video_classify.py
# python3 supported only 
#
# Video classification based on Inception-v3 & FCNN (Frames supported Convolution Network).
#
#
# 2017-06-22 by fengyoung(fengyoung1982@sina.com)
#

import sys
import os
import tensorflow as tf
import json
import time
import numpy as np
from embedding import util
from embedding import image_embedding
from embedding import fcnn
from embedding import vmp_file

tf.app.flags.DEFINE_string('graph_file', '', 'Inception-v3 graph file')
tf.app.flags.DEFINE_string('fcnn_model', '', 'FCNN model file')
tf.app.flags.DEFINE_string('input_file', '', 'Input video file')

tf.app.flags.DEFINE_integer('start_off', 0, 'Start offset in second. Default is 0')
tf.app.flags.DEFINE_integer('sfps', 1, 'Frames per second in sampling. Default is 1')

FLAGS = tf.app.flags.FLAGS


def extract_video_mat(inception_v3_graph_file, video_file, start_off = 0, sfps = 1):
	image_embedding.create_graph(inception_v3_graph_file)
	tf.logging.info(' Create graph from \"%s\"' % inception_v3_graph_file)

	with tf.Session() as sess:
		start_time = time.time()
		video_mat = image_embedding.feature_from_single_video_file(video_file, sess, start_off, sfps, fcnn.VM_HEIGHT, True)
	
	return video_mat


def classify(fcnn_model_file, video_mat): 
	fcnn_model = fcnn.FCNN()
	
	x = tf.placeholder(tf.float32, shape = [1, fcnn.VM_HEIGHT, fcnn.VM_WIDTH, 1])
	y = fcnn_model.classify(x)
	
	with tf.Session() as sess: 
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		saver.restore(sess, FLAGS.fcnn_model)
		tf.logging.info(' Restoring model from \"%s\"' % FLAGS.fcnn_model)
	
		vmat = np.reshape(util.mat_shape_normalize(video_mat, fcnn.VM_HEIGHT, fcnn.VM_WIDTH), [fcnn.VM_HEIGHT, fcnn.VM_WIDTH, 1])
		p = sess.run(y, {x: [vmat]})
		cate_prob = np.squeeze(p)
	
	return cate_prob 


def main(_):
	if not FLAGS.input_file:
		tf.logging.error(' The input file/path must be indicated!!')
		return -1
	if not os.path.exists(FLAGS.input_file): 	
		tf.logging.error(' The input file \"%s\" does\'t exist!!' % FLAGS.input_file)
		return -1
	
	if not FLAGS.graph_file: 
		tf.logging.error(' The inception-v3 graph file must be indicated!!')
		return -1
	if not os.path.exists(FLAGS.fcnn_model):
		tf.logging.error(' The inception-v3 graph file \"%s\" does\'t exist!' % FLAGS.graph_file)
		return -1
	
	if not FLAGS.fcnn_model: 
		tf.logging.error(' The FCNN model file must be indicated!!')
		return -1
	if not os.path.exists(FLAGS.fcnn_model):
		tf.logging.error(' The FCNN model file \"%s\" does\'t exist!' % FLAGS.fcnn_model)
		return -1

	video_mat = extract_video_mat(FLAGS.graph_file, FLAGS.input_file, FLAGS.start_off, FLAGS.sfps)
	cate_p = classify(FLAGS.fcnn_model, video_mat)	

	cate_id = list(cate_p).index(np.max(cate_p))
	print('---')
	for i in range(len(cate_p)): 
		if i == cate_id: 
			print('%d\t%.6g\t**' % (i, cate_p[i]))
		else: 
			print('%d\t%.6g' % (i, cate_p[i]))
	print('---')
	
	print('')
	return 0


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()


