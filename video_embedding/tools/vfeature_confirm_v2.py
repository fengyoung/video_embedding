# -*- coding: utf-8 -*- 
# file: vfeature_confirm.py 

import sys
sys.path.append("../")

import os
import functools
import time
import json
import numpy as np
import tensorflow as tf
from comm import util


def read_mid2cateid_file(mid2cateid_label_file): 
	try: 
		mid2cateid = {}	
		fp = open(mid2cateid_label_file, 'r')
		for line in fp: 
			if len(line.rstrip()) == 0: 
				continue
			s = line.rstrip().split('\t')
			mid2cateid[s[0]] = list(map(lambda x: int(x), s[1].split(',')))
		return mid2cateid	
	except IOError as err: 
		print("IO Error: %d" % err)
		return None
	finally: 
		fp.close()


def read_vfeature(path, filename):
	video_vecs = []	
	height = 0	
	width = 0
	try:
		fp = open(os.path.join(path, filename), 'r')
		for line in fp: 	
			if len(line.rstrip()) > 0: 
				vec = line.rstrip().split()[-1]
				if width == 0:
					width = len(vec.split(','))
				video_vecs.append(vec.replace(',', '_'))
				height += 1		
		return (height, width, video_vecs)
	except IOError as err:
		print("File Error: " + str(err))
		return None
	finally:	
		fp.close()


def read_vfeature_extend(path, filename):
	video_vecs = []	
	height = 0	
	width = 0
	try:
		fp = open(os.path.join(path, filename), 'r')
		for line in fp: 	
			if len(line.rstrip()) > 0: 
				vec = list(map(lambda x: float(x), line.rstrip().split()[-1].split(',')))
				if width == 0:
					width = len(vec) 
				video_vecs.append(vec)
				height += 1
		return (height, width, video_vecs)
	except IOError as err:
		print("File Error: " + str(err))
		return None
	finally:	
		fp.close()


def random_vmp_file_name(suffix):
	return 'vmp_' + time.strftime("%Y%m%d%H%M%S") + '_' + str(np.random.randint(10000, 99999)) + '.' + suffix
	

def confirm_to_pattern(vfeature_path, padding_to, mid2cateid, out_path, in_suffix = 'vfeature'): 
	"""
	mid,label0_label1_...._labelm,height_width,x0_x1_x2_..._xn
	"""
	if os.path.exists(out_path):
		print("Error: dir \"%s\" is exist!" % out_path)
		return False
	os.makedirs(out_path)

	num = 0
	cnt = 0
	fp = open(os.path.join(out_path, random_vmp_file_name('pattern')), 'w')
	for filename in util.get_filenames(vfeature_path, in_suffix): 
		mid = filename.split('.')[0].split('+')[-1]
		if mid in mid2cateid:
			height, width, video_vecs = read_vfeature(vfeature_path, filename)
			if len(video_vecs) == 0:
				print("(%d) mid: %s | ERROR: video_vecs is NULL" % (num + 1, mid))
			else:
				if (cnt + 1) % 64 == 0:
					fp.close()
					fp = open(os.path.join(out_path, random_vmp_file_name('pattern')), 'w')
				fp.write(mid)
				if len(mid2cateid[mid]) == 1:
					fp.write(',' + str(mid2cateid[mid][0]))
				else:
					fp.write(',' + functools.reduce(lambda x, y: str(x) + '_' + str(y), mid2cateid[mid]))
				fp.write(',' + str(padding_to) + '_' + str(width))
				if height < padding_to:
					fp.write(',' + functools.reduce(lambda x, y: str(x) + '_' + str(y), video_vecs))
					fp.write('_0' * ((padding_to - height) * width) + '\n')
					print("(%d) mid: %s | %d video_vecs, %d paddings" % (num + 1, mid, height, padding_to - height))
				else:
					fp.write(',' + functools.reduce(lambda x, y: str(x) + '_' + str(y), video_vecs[0:padding_to]) + '\n')
					print("(%d) mid: %s | %d video_vecs" % (num + 1, mid, height))
				cnt += 1
		else:
			print("(%d) mid: %s | ERROR: no cate labels" % (num + 1, mid))
		num += 1
	fp.close()
	return True



def confirm_to_tfrecord(vfeature_path, padding_to, mid2cateid, out_path, in_suffix = 'vfeature'): 
	if os.path.exists(out_path):
		print("Error: dir [%s] is exist!" % out_path)
		return False
	os.makedirs(out_path)

	num = 0
	num_rec = 0
	MIN_HEIGHT = 15

	writer = tf.python_io.TFRecordWriter(os.path.join(out_path, random_vmp_file_name('tfrecord')))
	for filename in util.get_filenames(vfeature_path, in_suffix): 
		mid = filename.split('.')[0].split('+')[-1]
		if mid in mid2cateid:
			height, width, video_vecs = read_vfeature_extend(vfeature_path, filename)
			if height < MIN_HEIGHT: 
				print("(%d) mid: %s | ERROR: video_vecs is NULL or too short" % (num + 1, mid))
			else: 
				off = 0
				while off + MIN_HEIGHT < height:
					if (num_rec + 1) % 128 == 0:
						writer.close()
						writer = tf.python_io.TFRecordWriter(os.path.join(out_path, random_vmp_file_name('tfrecord')))
					if off + padding_to < height:
						vecs = video_vecs[off:(off+padding_to)]
						print("(%d|%d) mid: %s, height: %d, off: %d | %d vecs" % (num + 1, num_rec + 1, mid, height, off, len(vecs)))
					else:	
						vecs = video_vecs[off:height] + [[0.] * width] * (off + padding_to - height)
						print("(%d|%d) mid: %s, height: %d, off: %d | %d vecs, %d paddings" % (num + 1, num_rec + 1, mid, height, off, len(vecs), off + padding_to - height))
					
					
					example = tf.train.Example(features = tf.train.Features(feature={
						"mid": tf.train.Feature(bytes_list = tf.train.BytesList(value = [str.encode(mid)])),
						"off": tf.train.Feature(int64_list = tf.train.Int64List(value = [off])),
						#"label": tf.train.Feature(int64_list = tf.train.Int64List(value = mid2cateid[mid])),
						"label": tf.train.Feature(bytes_list = tf.train.BytesList(value = [str.encode(util.number_list_to_string(mid2cateid[mid], '_'))])),
						"size": tf.train.Feature(int64_list = tf.train.Int64List(value = [padding_to, width])),   	
						"feature": tf.train.Feature(float_list = tf.train.FloatList(value = np.reshape(vecs, [-1])))
						}))
					writer.write(example.SerializeToString())
					off += MIN_HEIGHT
					num_rec += 1
		else:
			print("(%d|%d) mid: %s | ERROR: no cate labels" % (num + 1, num_rec + 1, mid))
		num += 1
	writer.close()
	return True


if __name__ == '__main__': 
	if len(sys.argv) != 6: 
		print("vfeature_confirm.py [--to_pattern <mid2cateid_label_file> <vfeature_path> <padding_to> <out_path>]")
		print("                    [--to_tfrecord <mid2cateid_label_file> <vfeature_path> <padding_to> <out_path>]")
		exit(-1)
	
	mid2cateid = read_mid2cateid_file(sys.argv[2])
	if not mid2cateid:
		print("Error: failed to load mid2cateid from \"%s\"" % sys.argv[2])
		exit(-1)

	if sys.argv[1] == "--to_pattern": 
		if not confirm_to_pattern(sys.argv[3], int(sys.argv[4]), mid2cateid, sys.argv[5]): 
			exit(-1)
	elif sys.argv[1] == "--to_tfrecord": 
		if not confirm_to_tfrecord(sys.argv[3], int(sys.argv[4]), mid2cateid, sys.argv[5]): 
			exit(-1)
	else:
		print("Error: \"%s\" is unsupported" % sys.argv[1])	
		exit(-1)

	exit(0)


