# -*- coding: utf-8 -*- 
# file: util.py 
# python2 & python3 supported 
# 
# 2017-04-03 by fengyoung(fengyoung1982@sina.com)
#

import os
import string
import time
import functools
import numpy as np


def get_filenames(path, suffix = None): 
	"""Get name of files in target path with special suffix.
	
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
	"""Get full path & name of files in target path with special suffix.

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


g_log_file = ""
def LOG(info_str, ltype = "INFO"):
	"""Output log string to g_log_file

	Args: 
		info_str: A `string`. The information for outputing
	    ltype: A `string`. Info-type which should be tagged in the front of log line
	"""
	if len(g_log_file) > 0:
		try:
			fp = open(g_log_file, 'a')
			fp.write("[" + ltype + "][" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "] " + info_str + "\n")
			return True
		except IOError as error:
			return False
		finally:
			fp.close()
	else:
		return True


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


def random_vmp_file_name(prefix, suffix):
	return prefix + '_' + time.strftime("%Y%m%d%H%M%S") + '_' + str(np.random.randint(10000, 99999)) + '.' + suffix
	


