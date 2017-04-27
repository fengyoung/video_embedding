# -*- coding: utf-8 -*- 
# file: collect_labels.py 
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



def collect_labels_from_filename(path, suffix, min_freq = 0): 
	cate_dict = {}
	tag_dict = {}
	cate_id = 0
	tag_id = 0
	for filename in util.get_filenames(path, suffix): 
		cates, tags, titles, mid = parse_tagged_filename(filename)
		for cate in cates: 
			if cate in cate_dict:
				cate_dict[cate][1] += 1
			else:
				cate_dict[cate] = [cate_id, 1]
				cate_id += 1
		for tag in tags: 
			if tag in tag_dict: 
				tag_dict[tag][1] += 1
			else: 
				tag_dict[tag] = [tag_id, 1]
				tag_id += 1
	
	rm_k = []
	for k, v in cate_dict.items(): 
		if v[1] < min_freq:
			rm_k.append(k)
	for k in rm_k:
		cate_dict.pop(k) 

	rm_k = []
	for k, v in tag_dict.items(): 
		if v[1] < min_freq:
			rm_k.append(k)
	for k in rm_k:
		tag_dict.pop(k)
	
	return (cate_dict, tag_dict)


def collect_labels_from_jsonfile(label_json_file, min_freq = 0): 
	cate_dict = {}
	tag_dict = {}
	cate_id = 0
	tag_id = 0
	try: 
		fp = open(label_json_file, 'r')
		for line in fp:
			if len(line.rstrip()) <= 0: 
				continue
			mid, json_str = line.rstrip().split('\t')
			d = json.loads(json_str)
			if "categories" in d: 
				for c in d["categories"]:
					if len(c["category"]) > 0: 
						if c["category"] in cate_dict: 
							cate_dict[c["category"]][1] += 1
						else:
							cate_dict[c["category"]] = [cate_id, 1]
							cate_id += 1
			if "tags" in d: 
				for t in d["tags"]:
					if len(t["tag"]) > 0: 
						if t["tag"] in tag_dict: 
							tag_dict[t["tag"]][1] += 1
						else:
							tag_dict[t["tag"]] = [tag_id, 1]
							tag_id += 1

		rm_k = []
		for k, v in cate_dict.items(): 
			if v[1] < min_freq:
				rm_k.append(k)
		for k in rm_k:
			cate_dict.pop(k) 

		rm_k = []
		for k, v in tag_dict.items(): 
			if v[1] < min_freq:
				rm_k.append(k)
		for k in rm_k:
			tag_dict.pop(k)

		return (cate_dict, tag_dict)
	except IOError as err:
		print("IO Error: " + str(err))			
		return False
	finally:
		fp.close()



def save_labels(label_dict, label_file): 
	try:
		fp = open(label_file, 'w')
		for label in label_dict: 
			fp.write(str(label_dict[label][0]) + '\t' + label + '\t' + str(label_dict[label][1]) + '\n')
		return True
	except IOError as err:
		print("File Error: " + str(err))
		return False
	finally:
		fp.close()	


if __name__ == '__main__': 
	if len(sys.argv) != 5: 
		print("collect_labels.py [--from_filename <data_path> <min_freq> <out_path>]")
		print("                  [--from_jsonfile <json_file> <min_freq> <out_path>]") 
		exit(-1)
	
	if os.path.exists(sys.argv[4]):
		print("Error: dir [%s] is exist!" % sys.argv[4])
		exit(-1)
	os.makedirs(sys.argv[4])

	if sys.argv[1] == "--from_filename": 
		cate_dict, tag_dict = collect_labels_from_filename(sys.argv[2], "vfeature", int(sys.argv[3])) 
	elif sys.argv[1] == "--from_jsonfile": 
		cate_dict, tag_dict = collect_labels_from_jsonfile(sys.argv[2], int(sys.argv[3]))
	else:
		print("Error: unsupported \"%s\"" % sys.argv[1])	
		eixt(-1) 
	
	save_labels(cate_dict, os.path.join(sys.argv[4], "cate_id.map"))
	save_labels(tag_dict, os.path.join(sys.argv[4], "tag_id.map"))

	exit(0)


