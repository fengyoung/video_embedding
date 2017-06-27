# -*- coding: utf-8 -*- 
# file: util.py 
# python3 supported only 
# 
# 2017-04-03 by fengyoung(fengyoung1982@sina.com)
#

import os
import string
import time
import functools
import numpy as np


def get_filenames(path, suffix = None): 
	"""Gets name of files in target path with special suffix.
	
	Args: 
		path: A `string`. Target path.
		suffix: A `string`. Special suffix. If None, all files should be output
	
	Returns:
		A list of file names in string.
	"""
	filenames = []
	for filename in os.listdir(path):
		if suffix:
			if filename.split(".")[-1] == suffix: 
				filenames.append(filename)
		else: 
			filenames.append(filename)
	return filenames


def get_filepaths(path, suffix = None): 
	"""Gets full path & name of files in target path with special suffix.

	Args: 
		path: A `string`. Target path.
		suffix: A `string`. Special suffix. If None, all files should be output

	Returns:
	    A list of file path & names in string.
	"""
	filepaths = []
	for filename in os.listdir(path):
		if suffix: 
			if filename.split(".")[-1] == suffix: 
				filepaths.append(os.path.join(path, filename))
		else: 
			filepaths.append(os.path.join(path, filename))
	return filepaths


def number_list_to_string(num_list, delimiter = ','):  
	"""Converts number list to a string splitted by special delimiter

	Args: 
		num_list: A list of numbers. 
	    delimiter: A `string`. delimiter charactors

	Returns:
		String which contains the number list elements
	"""
	if len(num_list) == 0:
		return ""
	elif len(num_list) == 1:
		return str(num_list[0])
	else:
		return functools.reduce(lambda x, y: str(x) + delimiter + str(y), num_list)


def string_to_integer_list(sstr, delimiter = ','):
	"""Converts a string which splitted by special delimiter to integer list 

	Args: 
		sstr: String which contains the number list elements
	    delimiter: A `string`. delimiter charactors

	Returns:
		Integer list
	"""
	ss = sstr.split(delimiter)
	if len(ss) == 0:
		return []
	elif len(ss) == 1:
		return [int(ss[0])]
	else:
		return list(map(lambda x: int(x), ss)) 

	
def string_to_float_list(sstr, delimiter = ','):
	"""Converts a string which splitted by special delimiter to float list 

	Args: 
		sstr: String which contains the number list elements
	    delimiter: A `string`. delimiter charactors

	Returns:
		Float list
	"""
	ss = sstr.split(delimiter)
	if len(ss) == 0:
		return []
	elif len(ss) == 1:
		return [float(ss[0])]
	else:
		return list(map(lambda x: float(x), ss)) 


def one_hot_to_sparse(one_hot): 
	"""Converts one hot list to sparse list. 

	Args: 
		one_hot: A list of int/float. One hot list 

	Returns:
		Sparse list of index 
	"""
	sparse = []
	for i in range(len(one_hot)): 
		if int(one_hot[i]) == 1:
			sparse.append(i)
	return sparse


def random_vmp_file_name(prefix, suffix):
	"""Generates vmp file name randomly. 

	Args: 
		prefix: A string. Prefix of the file name
		suffix: A string. Suffix of the file name

	Returns:
		New name of vmp file
	"""
	return prefix + '_' + time.strftime("%Y%m%d%H%M%S") + '_' + str(np.random.randint(10000, 99999)) + '.' + suffix


def mat_shape_normalize(mat, t_height, t_width): 
	m_shape = mat.shape
	if t_height == m_shape[0] and t_width == m_shape[1]: 
		return mat
	elif t_width == m_shape[1]:
		return np.array(list(mat) + list(np.zeros([t_height - m_shape[0], t_width])))
	else:
		new_mat = []
		if m_shape[1] < t_width: 
			for r in range(np.min([t_height, m_shape[0]])):
				new_mat.append(list(mat[r]) + [0.] * (t_width - m_shape[1]))			
		else: 
			for r in range(np.min([t_height, m_shape[0]])):
				new_mat.append(list(mat[r][0:t_width]))
		if len(new_mat) < t_height:
			return np.array(new_mat + list(np.zeros([t_height - len(new_mat), t_width])))
		else:
			return np.array(new_mat)







