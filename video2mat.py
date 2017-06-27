# -*- coding: utf-8 -*- 
# file: video2mat.py
# python3 supported only 
#
# Extracts 2-D features from video(s) and saved to VMP file(s). The extraction is based on Inception-v3.
# Features of a video should be extracted as 2-D video-matrix, which can be used representing the content of the video
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
from embedding import vmp_file

tf.app.flags.DEFINE_string('graph_file', '', 'Inception-v3 graph file')
tf.app.flags.DEFINE_string('input', '', 'Input video file/path')
tf.app.flags.DEFINE_string('output', '', 'Output file/path. ')

tf.app.flags.DEFINE_string('label_file', '', 'Label tagged file in json format. Optional')
tf.app.flags.DEFINE_string('out_fmt', 'pattern-string', 'Output format. \"pattern-string\" or \"tfrecord\". Default is \"pattern-string\"')
tf.app.flags.DEFINE_integer('start_off', 0, 'Start offset in second. Default is 0')
tf.app.flags.DEFINE_integer('sfps', 1, 'Frames per second in sampling. Default is 1')
tf.app.flags.DEFINE_integer('max_frames', 60, 'How many frames shouled be extracted, Default is 60')
tf.app.flags.DEFINE_boolean('padding', True, 'If 0 should be padded when the video was too short, Default is True')

FLAGS = tf.app.flags.FLAGS
LABEL_DICT = {}

def read_label_tag_from_jsonfile(label_tag_file):
	fp = open(label_tag_file, 'r')
	try:
		global LABEL_DICT
		LABEL_DICT = {}
		for line in fp:
			if len(line.rstrip()) == 0:
				continue
			d = json.loads(line.rstrip())
			LABEL_DICT[d["file_name"]] = [d["mid"], d["label"]]
		return True
	except IOError as err:
		return False
	finally:
		fp.close()


def main(_):
	if not FLAGS.input:
		tf.logging.error(' The input file/path must be indicated!!')
		return -1
	
	if not FLAGS.output:
		tf.logging.error(' The output file/path must be indicated!!')
		return -1
	if os.path.isdir(FLAGS.output):
		tf.logging.error(' \"%s\" is existing as dir!!' % FLAGS.output)
		return -1
	
	if FLAGS.out_fmt != 'tfrecord' and FLAGS.out_fmt != 'pattern-string':
		tf.logging.error(' Unsupported out format %s' % FLAGS.out_fmt)
		return -1

	# load labels if exists 
	if FLAGS.label_file:
		if not read_label_tag_from_jsonfile(FLAGS.label_file):
			tf.logging.error(' Failed to load labels from %s' % FLAGS.label_file)
			return -1
	
	# create the graph from model file
	if not FLAGS.graph_file: 
		tf.logging.error(' The graph file must be indicated!!')
		return -1
	image_embedding.create_graph(FLAGS.graph_file)

	with tf.Session() as sess:
		mids = []
		labels = []
		video_mats = []
		if os.path.isfile(FLAGS.input):
			start_time = time.time()
			video_features = image_embedding.feature_from_single_video_file(FLAGS.input, sess, FLAGS.start_off, FLAGS.sfps, FLAGS.max_frames, FLAGS.padding)
			video_mats.append(video_features)
			filename = os.path.split(FLAGS.input)[-1]	
			if filename in LABEL_DICT:
				mids.append(LABEL_DICT[filename][0])
				labels.append(LABEL_DICT[filename][1])
			else:
				mids.append('xxxxxxxx')
				labels.append([0])
			if FLAGS.out_fmt == 'pattern-string':
				vmp_file.write_as_pattern_string(mids, labels, np.array(video_mats), FLAGS.output) 
				tf.logging.info(' Video-Mat has been extracted and saved as pattern-string proto. Time cost (sec): %f\nPlease check %s' % (time.time() - start_time, FLAGS.output))
			else:
				vmp_file.write_as_tfrecord(mids, labels, np.array(video_mats), FLAGS.output, sess = sess) 
				tf.logging.info(' Video-Mat has been extracted and saved as tfrecord proto. Time cost (sec): %f\nPlease check %s' % (time.time() - start_time, FLAGS.output))
		else: 
			video_files = util.get_filepaths(FLAGS.input)
			file_id = 1
			os.makedirs(FLAGS.output)
			start_time = time.time()
			for video_file in video_files:
				video_features = image_embedding.feature_from_single_video_file(video_file, sess, FLAGS.start_off, FLAGS.sfps, FLAGS.max_frames, FLAGS.padding)
				video_mats.append(video_features)
				filename = os.path.split(video_file)[-1]
				if filename in LABEL_DICT:
					mids.append(LABEL_DICT[filename][0])
					labels.append(LABEL_DICT[filename][1])
				else:
					mids.append('xxxxxxxx')
					labels.append([0])
				if len(mids) == 128:
					if FLAGS.out_fmt == 'pattern-string':
						out_file = os.path.join(FLAGS.output, 'video_mats_%06d.pattern' % file_id)
						vmp_file.write_as_pattern_string(mids, labels, np.array(video_mats), out_file)
						tf.logging.info(' %d Video-Mat(s) has been extracted and saved as pattern-string proto. Time cost (sec): %f\nPlease check %s' % (len(mids), time.time() - start_time, out_file))
					else: 
						out_file = os.path.join(FLAGS.output, 'video_mats_%06d.tfrecord' % file_id)
						vmp_file.write_as_tfrecord(mids, labels, np.array(video_mats), out_file, sess = sess)
						tf.logging.info(' %d Video-Mat(s) has been extracted and saved as tfrecord proto. Time cost (sec): %f\nPlease check %s' % (len(mids), time.time() - start_time, out_file))
					mids.clear()
					labels.clear()
					video_mats.clear()
					file_id += 1
					start_time = time.time()
			if len(mids) > 0: 
				if FLAGS.out_fmt == 'pattern-string':
					out_file = os.path.join(FLAGS.output, 'video_mats_%06d.pattern' % file_id)
					vmp_file.write_as_pattern_string(mids, labels, np.array(video_mats), out_file)
					tf.logging.info(' %d Video-Mat(s) has been extracted and saved as pattern-string proto. Time cost (sec): %f\nPlease check %s' % (len(mids), time.time() - start_time, out_file))
				else: 
					out_file = os.path.join(FLAGS.output, 'video_mats_%06d.tfrecord' % file_id)
					vmp_file.write_as_tfrecord(mids, labels, np.array(video_mats), out_file, sess = sess)
					tf.logging.info(' %d Video-Mat(s) has been extracted and saved as tfrecord proto. Time cost (sec): %f\nPlease check %s' % (len(mids), time.time() - start_time, out_file))

	return 0


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()


