# -*- coding: utf-8 -*- 
# file: video_decode.py 
# python3 supported only
# 
# Video decoding based on OpenCV. 
# 
# 2017-06-22 by fengyoung(fengyoung1982@sina.com)
#

import sys
sys.path.append("../")

import os
import cv2
import numpy as np


def frames_extract(video_file, start_off = 0, sampling_fps = 1, max_frame_cnt = 60): 
	"""Extracts frames input video.
	
	Args: 
		video_file: A String. Video file path.
		start_off: A Integer`. Start offset in second. 
		sampling_fps: A Integer. Frames per second in sampling. 
		max_frame_cnt: A Integer. How many frames should be extracted.
		padding: A Boolean. If 0 should be padded when the video was too short.

	Returns:
		A 4-D array of uint8 in shape [frames, heigt, width, channels]. If failed, return None 
	"""
	# open the video and get its FPS & size 
	if not os.path.exists(video_file):	
		return None

	vcap = cv2.VideoCapture(video_file)
	fps = vcap.get(cv2.CAP_PROP_FPS)
	height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
	width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)) 

	frame_interval = 1
	if sampling_fps > 0 and sampling_fps < fps:
		frame_interval = int(fps / sampling_fps)
	frame_list = []
	off = 0
	cnt = 0

	# extract frames	
	success, im = vcap.read()
	while success and cnt < max_frame_cnt: 
		if off == start_off * fps + cnt * frame_interval:
			frame_list.append(im)
			cnt += 1
		off += 1
		success, im = vcap.read()
	
	return np.array(frame_list)	



