# -*- coding: utf-8 -*- 
# file: cate_result_extract.py

import sys


def read_modified_cateid_map(modified_cateid_map_file): 
	try:
		fp = open(modified_cateid_map_file, 'r')
		id_to_cate = {}
		for line in fp:
			if len(line.rstrip()) == 0:
				continue
			k, v = line.rstrip().split('\t')
			id_to_cate['[' + k + ']'] = v.split(',')[0]
		return id_to_cate
	except IOError as err:
		print("IO Error %d" % err)
		return None
	finally:
		fp.close()


def parse_result_record(res_rec, id_to_cate):
	if res_rec.find('mid') < 0:
		return None
	v_str = res_rec.split('|')[1].rstrip().lstrip()	
	mid = v_str.split(',')[0].split(':')[1].lstrip().rstrip()
	cateid_0, cateid_1 = v_str.split(',')[1].split('->')
	cate_0 = id_to_cate[cateid_0.rstrip().lstrip()]
	cate_1 = id_to_cate[cateid_1.rstrip().lstrip()]
	return (mid, cate_0, cate_1)


if __name__ == "__main__":
	if len(sys.argv) != 3: 
		print("usage: %s <result_file> <modified_cateid_map_file>" % sys.argv[0])
		exit(-1)

	id_to_cate = read_modified_cateid_map(sys.argv[2])
	if not id_to_cate: 
		print("Error: failed to read cate_id map")
		exit(-1)

	try:
		fp = open(sys.argv[1])
		for line in fp:
			if len(line.rstrip()) == 0:
				continue
			ret = parse_result_record(line.rstrip(), id_to_cate)
			if ret:
				mid, cate_0, cate_1 = ret
				print("%s\t%s\t%s" % (mid, cate_0, cate_1))	
	except IOError as err:
		print("IO Error %d" % err)
		exit(-1)
	finally:
		fp.close()



	exit(0)

