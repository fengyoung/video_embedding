# -*- coding: utf-8 -*- 
# file: vmat2vec.py
# python3 supported only 
#
# Frames embeding from 2-D video-mat(s) to 1-D video-vec(s). 
# This processing of Frames Embedding is based on FCNN (Frames supported Convolution Network)
# Features of a video should be extracted as 
# The 1-D video-vec can be used representing the content of the video
#
#
# 2017-06-22 by fengyoung(fengyoung1982@sina.com)
#

import sys
import os
import tensorflow as tf
import time
import numpy as np
from embedding import util
from embedding import fcnn
from embedding import vmp_file

tf.app.flags.DEFINE_string('input_file', '', 'Input vmp file. Pattern-string proto supported only')
tf.app.flags.DEFINE_string('fcnn_model', '', 'FCNN model file')
tf.app.flags.DEFINE_string('output_file', '', 'Output video-vec file. Pattern-string proto supported only')

tf.app.flags.DEFINE_integer('batch_size', '1', 'Default is 1')

FLAGS = tf.app.flags.FLAGS


def main(_):
	if not FLAGS.input_file:
		tf.logging.error(' The input file/path must be indicated!!')
		return -1
	if not os.path.exists(FLAGS.input_file): 
		tf.logging.error(' The input file \"%s\" does\'t exist!!' % FLAGS.input_file)
		return -1

	if not FLAGS.output_file:
		tf.logging.error(' The output file/path must be indicated!!')
		return -1

	if not FLAGS.fcnn_model: 
		tf.logging.error(' The FCNN model file must be indicated!!')
		return -1
	if not os.path.exists(FLAGS.fcnn_model):
		tf.logging.error(' The FCNN model file \"%s\" does\'t exist!' % FLAGS.fcnn_model)
		return -1
	
	fcnn_model = fcnn.FCNN()
	global_step = tf.Variable(0, name="global_step", trainable=False)
	append = False

	with tf.Session() as sess: 
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		saver.restore(sess, FLAGS.fcnn_model)
		tf.logging.info(' Restoring model from \"%s\"' % FLAGS.fcnn_model)

		x = tf.placeholder(tf.float32, shape = [1, fcnn.VM_HEIGHT, fcnn.VM_WIDTH, 1])
		z = fcnn_model.feature_detect(x)

		fp = open(FLAGS.input_file, 'r')
		try:
			mids = []
			labels = []
			video_vecs = []
			cnt = 0
			for line in fp: 
				if len(line.rstrip()) == 0:
					continue
				start_time = time.time()
				mid, label, feat = vmp_file.parse_pattern_string(line.rstrip())
				video_mat = np.reshape(util.mat_shape_normalize(feat, fcnn.VM_HEIGHT, fcnn.VM_WIDTH), [fcnn.VM_HEIGHT, fcnn.VM_WIDTH, 1])
				vvec = sess.run(z, {x: [video_mat]})
				video_vec = np.squeeze(vvec)
				mids.append(mid)	
				labels.append(label)	
				video_vecs.append(video_vec)
				cnt += 1
				tf.logging.info(' (%d) %s has been processed. Time cost (sec): %f' % (cnt, mid, time.time() - start_time))
				start_time = time.time()
				if len(mids) == 10: 
					vmp_file.video_vec_write_as_pattern_string(mids, labels, np.array(video_vecs), FLAGS.output_file, append = append)
					append = True
					mids.clear()
					labels.clear()
					video_vecs.clear()
			if len(mids) > 0: 
				vmp_file.video_vec_write_as_pattern_string(mids, labels, np.array(video_vecs), FLAGS.output_file, append = append)
		except IOError as err:
			tf.logging.warn(' Read \"%s\" error: %d' % (vmp_file, err))
		finally:
			fp.close()
			
	return 0


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()


