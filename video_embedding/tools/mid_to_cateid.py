# -*- coding: utf-8 -*- 
# file: mid_to_cateid.py 
#

import sys
sys.path.append("../")
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import functools
import time
import numpy as np
from comm import util
import json
import codecs


def load_label_dict(label_map_file):
	try:
		num_labels = 0
		label_dict = {}
		fp = open(label_map_file, 'r')
		for line in fp:
			if len(line.rstrip()) > 0: 
				s = line.rstrip().split('\t')
				label_id = int(s[0])
				if num_labels <= label_id:
					num_labels = label_id + 1
				for label in s[1].split(','):
					label_dict[label.decode('utf-8')] = label_id
		return (num_labels, label_dict)	
	except IOError as err:
		print("IO Error: " + str(err))			
		return None
	finally:
		fp.close()


def parse_tagged_filename(filename): 
	cates = []
	tags = []
	titles = []
	mid = ''
	for seg in filename.split('.')[0].split('+'): 
		s = seg.split('_')
		if len(s) == 1:
			mid = seg
		else:
			if s[0] == 'cates' and len(s[1]) > 0:
				cates = s[1:]
			elif s[0] == 'tags' and len(s[1]) > 0:
				tags = s[1:]
			elif s[0] == 'titles' and len(s[1]) > 0:
				titles = s[1:]
	return (cates, tags, titles, mid)
 

def label_encode(labels, label_dict, num_labels):
	#ids = [0] * num_labels
	ids = []
	if len(labels) == 0:
		return None
	else:
		for label in labels:
			if label.decode('utf-8') in label_dict: 
				#ids[label_dict[label.decode('utf-8')]] = 1
				ids.append(label_dict[label.decode('utf-8')]) 
	if len(ids) == 0:
		return None
	return ids


def mid2cateid_from_jsonfile(label_json_file, cate_dict, cate_num): 
	mid2cateid = {}
	try: 
		fp = open(label_json_file, 'r')
		#fp = codecs.open(label_json_file, 'r', encoding = 'utf-8')
		for line in fp:
			if len(line.rstrip()) <= 0: 
				continue
			mid, json_str = line.rstrip().split('\t')
			d = json.loads(json_str)
			if "categories" in d:
				cates = []
				for c in d["categories"]:
						cates.append(c["category"])
			cate_id = label_encode(cates, cate_dict, cate_num)
			if cate_id: 
				mid2cateid[mid] = cate_id
		return mid2cateid
	
	except IOError as err:
		print("IO Error: " + str(err))			
		return None
	finally:
		fp.close()


def mid2cateid_from_filename(path, cate_dict, cate_num, suffix): 
	mid2cateid = {}
	for filename in util.get_filenames(path, suffix): 
		cates, tags, titles, mid = parse_tagged_filename(filename)
		cate_id = label_encode(cates, cate_dict, cate_num)
		if cate_id: 
			mid2cateid[mid] = cate_id
	return mid2cateid


def save_mid2cateid(mid2cateid, out_file):
	try:
		fp = open(out_file, 'w')
		for k,v in mid2cateid.items():
			if len(v) == 1: 
				fp.write(str(k) + '\t' + str(v[0]) + '\n')
			else: 
				fp.write(str(k) + '\t' + functools.reduce(lambda x, y: str(x) + ',' + str(y), v) + '\n')
		return True	
	except IOError as err:
		print("IO Error: " + str(err))			
		return False
	finally:
		fp.close()



if __name__ == '__main__':
	if len(sys.argv) != 5: 
		print("mid_to_cateid.py [--from_filename <data_path> <modified_cate_map_file> <out_file>]")
		print("                 [--from_jsonfile <json_file> <modified_cate_map_file> <out_file>]") 
		exit(-1)

	
	ret = load_label_dict(sys.argv[3])
	if not ret:
		print("Error: failed to load cate-mapping from %s" % sys.argv[3])
		exit(-1)
	cate_num, cate_dict = ret

	
	if sys.argv[1] == "--from_filename": 
		mid2cateid = mid2cateid_from_filename(sys.argv[2], cate_dict, cate_num, "vfeature") 
	elif sys.argv[1] == "--from_jsonfile": 
		mid2cateid = mid2cateid_from_jsonfile(sys.argv[2], cate_dict, cate_num) 
	else:
		print("Error: \"%s\" is unsupported" % sys.argv[1])	
		exit(-1)	

	if not save_mid2cateid(mid2cateid, sys.argv[4]):
		print("Error: failed to save resutl to \"%s\" is unsupported" % sys.argv[4])	
		exit(-1)

	exit(0)


