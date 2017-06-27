# -*- coding: utf-8 -*- 
# file: train_fcnn.py
# python3 supported only 
#
# Train a Frames supported Convolution Network (FCNN). 
#
#
# 2017-06-22 by fengyoung(fengyoung1982@sina.com)
#

import sys
import os
import time
import tensorflow as tf
from embedding import util
from embedding import vmp_file
from embedding import fcnn


tf.app.flags.DEFINE_string('vmp_path', '', 'Training VMP data path. Tfrecord proto supported only')
tf.app.flags.DEFINE_string('naming', '', 'The name of this model. Determine the path to save checkpoint and events file.')
tf.app.flags.DEFINE_string('model_path', '', 'Root path to save checkpoint and events file. The final path would be <model_path>/<naming>')

tf.app.flags.DEFINE_integer('epoch', 100, 'Default is 100')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Default is 50')

FLAGS = tf.app.flags.FLAGS

def main(_):
	# make sure the training path exists.
	training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
	if not os.path.exists(training_path): 
		os.makedirs(training_path)

	# get vmpattern_files queue
	vmp_files = util.get_filepaths(FLAGS.vmp_path)
	if len(vmp_files) == 0:
		tf.logging.fatal(" Error: path \"%s\" is empty or not exist!" % FLAGS.vmp_path) 
		return -1

	# prepare to read VMP
	mid_batch, y0_batch, x_batch = vmp_file.prepare_read_from_tfrecord(vmp_files, fcnn.NUM_LABELS, fcnn.VM_HEIGHT, fcnn.VM_WIDTH, batch_size = FLAGS.batch_size, max_epochs = FLAGS.epoch, num_threads = 12, shuffle = True) 
	
	# create the FCNN model
	fcnn_model = fcnn.FCNN()

	# prepare to train
	global_step = tf.Variable(0, name="global_step", trainable=False)
	y_batch = fcnn_model.propagate_to_classifier(x_batch)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y0_batch, logits = y_batch))
	train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step = global_step)
	
#	correct_prediction = tf.equal(tf.argmax(y_batch, 1), tf.argmax(y0_batch, 1))
#	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess: 
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# Restore variables for training model if the checkpoint file exists.
		last_file = tf.train.latest_checkpoint(training_path)
		if last_file:
			tf.logging.info(' Restoring model from \"%s\"' % last_file)
			saver.restore(sess, last_file)
	
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		start_time = time.time()
		b_time = time.time()
		try:
			while not coord.should_stop():
				_, loss_t, step = sess.run([train_op, loss, global_step])
				elapsed_time = time.time() - start_time
				start_time = time.time()
				"""logging"""
				if step % 10 == 0:
					tf.logging.info(' step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
				"""checkpoint"""
				if step % 100 == 0:
					ckpt_file = os.path.join(training_path, fcnn.CPKT_FILE_NAME)
					saver.save(sess, ckpt_file, global_step = step)
					tf.logging.info(' Save checkpoint to \"%s-%d\".' % (ckpt_file, step))
		except tf.errors.OutOfRangeError:
			ckpt_file = os.path.join(training_path, fcnn.CPKT_FILE_NAME + '-done')
			saver.save(sess, ckpt_file) 
			total_time = time.time() - b_time
			m, s = divmod(int(total_time), 60)
			h, m = divmod(m, 60)
			tf.logging.info(' Done training -- epoch limit reached. Time Cost: %02d:%02d:%02d' % (h, m, s))
			tf.logging.info(' Please copy \"%s\" as the final model.' % ckpt_file)
		finally:
			coord.request_stop()

		coord.join(threads)
	return 0


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()



