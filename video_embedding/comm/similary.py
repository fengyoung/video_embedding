# -*- coding: utf-8 -*- 
# file: similary.py 
# python2 & python3 supported 
# 
# 2017-04-03 by fengyoung(fengyoung1982@sina.com)
#

import numpy as np


def cosine_sim(v1, v2):
	"""Similary based on Consine

	Arg: 
		v1/v2: a list of float. feature vector

	Returns:
		The value of similary
	"""
	if len(v1) != len(v2):
		return False
	cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	return 0.5 + 0.5 * cos


def euclidean_sim(v1, v2):
	"""Similary based on Euclidean Distance. 

	Arg: 
		v1/v2: a list of float. feature vector

	Returns:
		The value of similary
	"""
	if len(v1) != len(v2):
		return False
	euc_dist = np.linalg.norm(v1 - v2)
	return 1.0 / (1.0 + euc_dist)

	 
def manhattan_sim(v1, v2):
	"""Manhattan Similary

	Arg: 
		v1/v2: a list of float. feature vector

	Returns:
		The value of similary
	"""
	if len(v1) != len(v2):
		return False
	manh_dist = np.sum(np.abs(v1 - v2))
	return 1.0 / (1.0 + manh_dist)


def chebyshev_sim(v1, v2):
	"""Chebyshev Similary

	Arg: 
		v1/v2: a list of float. feature vector

	Returns:
		The value of similary
	"""
	if len(v1) != len(v2):
		return False
	cheby_dist = np.max(np.abs(v1 - v2))
	return 1.0 / (1.0 + cheby_dist)



