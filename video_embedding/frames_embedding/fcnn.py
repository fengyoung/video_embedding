# -*- coding: utf-8 -*- 
# file: fcnn.py 
# python3 supported only 
# 
# Frames supported Convolution Neural Network (FCNN)
# 
# There are 4 classes defined:  
#	FrameConvLayer: Frame supported Convolutional Layer. 
#	|- activate(): Activates current frames supported convolutional layer. 
# 
#	DenseConnLayer: Densely Connect Layer.
#	|- activate(): Activates current densely connect layer. 
# 
#	SoftmaxLayer: Softmax Layer for multi-classification.
#	|- propagate_to(): Propagate to the layer but don't activated.
#	|- activate(): Activates current Softmax Layer.
# 
#	FCNN: Frame Convolution Neural Network
#	|- create(): Create FCNN according to input or internal architecture dict
#	|- propagate_to_classifier(): Propagate to the last layer but don't activated by softmax.
#	|- classify(): Feedforward from input to output, includes activation of softmax for classification. 
#	|- feature_detect(): Feedforward from input to the last hidden layer. This operation could be used as feature detection.
#	|- save_arch(): Save FCNN architecture parameters in json format to file
#	|- read_arch(): Read FCNN architecture parameters in json format from file 
# 
# 2017-04-18 by fengyoung(fengyoung1982@sina.com)
#

import tensorflow as tf
import numpy as np
import json
from functools import reduce


g_model_file_name = 'video2vec_fcnn.cpkt'
g_arch_file_name = 'fcnn_arch.json'


def weight_variable(shape, name = None): 
	"""Initialize weights conform normal distribution

	Args: 
		shape: A list of ints. Shape of weight tensor

	Returns:
		A tensor of the input shape
	"""
	# intitalize the weight
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name = name)


def bias_variable(shape, name = None): 
	"""Initialize bias

	Args: 
		shape: A list of ints. Shape of bias 

	Returns:
		A tensor of the input shape
	"""
	initial = tf.constant(0.1, shape = shape) 
	return tf.Variable(initial, name = name)


class FrameConvLayer: 
	""" Frame supported Convolutional Layer.
	"""
	def __init__(self, w_shape, conv_h_stride, layer_id, pool_params = None): 
		"""Constructs a new Frame supported Convolution Layer.

		Args:
			w_shape: A list of ints. Shape of weights as [height, 1, in_channels, out_channels]. 
			conv_h_stride: An `integer`. The horizontal stride of the sliding window for each dimension in convolution.
			layer_id: An `integer` ID of current layer.
			pool_params: A list of ints. [k_size, hori_stride].
		""" 
		self.w = weight_variable(w_shape, name = 'LAYER' + str(layer_id) + '_frame_conv_w') 
		self.bias = bias_variable([w_shape[3]], name = 'LAYER' + str(layer_id) + '_frame_conv_bias') 
		self.conv_h_stride = conv_h_stride
		self.pool_params = None
		if pool_params:	
			self.pool_params = pool_params[:]
	
	def activate(self, x): 
		"""Activates current Frame Convolution Layer. 

		There are 3 steps as following: 
		1. Computes the Frame Convolution between input (x) and filter (self.w).   
		2. Activates the convolution result by ReLU. 
		3. Performs max-pooling if necessary.

		Args: 
			x: A 4-D tensor of shape [batch, height, width, channels] and type float32. The input of convolution.
	
		Important:
			The channels of input must equals to in_channel of the filter (self.w).
	
		Returns: 
			A 4-D tensor of the same shape and same type of input. The output of current layer.
		"""
		# frame convolution and ReLU activation
		h_conv = tf.nn.relu(tf.nn.conv2d(x, self.w, strides = [1, self.conv_h_stride, 1, 1], padding = 'SAME') + self.bias)
		# max pooling	
		if self.pool_params: 
			return tf.nn.max_pool(h_conv, ksize = [1, self.pool_params[0], 1, 1], strides = [1, self.pool_params[1], 1, 1], padding = 'SAME')
		else:
			return h_conv


class DenseConnLayer:
	"""Densely Connect Layer.
	"""
	def __init__(self, w_shape, layer_id):
		"""Constructs a new Densely Connect Layer.

		Args:
			w_shape: A list of ints. Shape of weights as [input_size, output_size]. 
			layer_id: An `integer` ID of current layer.
		""" 
		self.w = weight_variable(w_shape, name = 'LAYER' + str(layer_id) + '_dense_conn_w')
		self.bias = bias_variable([w_shape[1]], name = 'LAYER' + str(layer_id) + '_dense_conn_bias')
	
	def activate(self, x):
		"""Activates current Densely Connect Layer.

		There are 2 steps as following: 
		1. Reshape the input tensor into a batch of vectors.
		2. Multiply by weight matrix, add a bias, and apply a ReLU. 

		Args:
			x: A 4-D tensor of shape [batch, height, width, channels] and type float32. The input of layer.

		Important: 
			Because the tensor would be stretch to vectors, the size of them should satisfies (height * width * channels) = input_size 
	
		Returns: 
			A 2-D tensor of shape [batch, output_size] and the same type of input. The output of current layer.
		"""
		# reshape the input into a batch of vectors
		h_flat = tf.reshape(x, [tf.shape(x)[0], -1])
		# propagation & activated by ReLU
		return tf.nn.relu(tf.matmul(h_flat, self.w) + self.bias)
		 

class SoftmaxLayer: 
	"""Softmax Layer for multi-classification.
	"""
	def __init__(self, w_shape, layer_id): 
		"""Constructs a new Softmax Layer.
		
		Args:
			w_shape: A list of ints. Shape of weights as [input_size, output_size]. 
			layer_id: An `integer` ID of current layer.
		"""
		self.w = weight_variable(w_shape, name = 'LAYER' + str(layer_id) + 'softmax_w') 	
		self.bias = bias_variable([w_shape[1]], name = 'LAYER' + str(layer_id) + 'softmax_bias') 	
	
	def propagate_to(self, x):
		"""Propagate to the layer but don't activated.
		
		Args: 
			x: A 2-D tensor of shape [batch, input_size] and type float32. The input of layer
	
		Returns: 
			A 2-D tensor of shape [batch, output_size] and the same type of input. No-actived output of current layer.
		"""
		return tf.matmul(x, self.w) + self.bias

	def activate(self, x):
		"""Activates current Softmax Layer.

		Args: 
			x: A 2-D tensor of shape [batch, input_size] and type float32. The input of layer

		Returns: 
			A 2-D tensor of shape [batch, output_size] and the same type of input. The output of current layer.
		""" 
		return tf.nn.softmax(tf.matmul(x, self.w) + self.bias)


class FCNN:
	"""Frame Convolution Neural Network
	"""
	def __init__(self, arch_dict = None): 
		"""Constructs a new FCNN

		Args:
			arch_dict: A dict. The mapping of the network architecture
		"""
		if arch_dict:
			self.arch = arch_dict.copy()
		else:
			self.arch = None
		self.create()

	def create(self, arch_dict = None): 
		"""Create FCNN according to input or internal architecture dict
		
		Args:
			arch_dict: the architecture dict. If None, the internal arch should be used
		
		The proto of arch_dict is
		--------------------------------------------------------------------------------------
		{
		  "in_height": 30, 
		  "in_width": 2048,
		  "frame_conv_layers": 
		  [
		    {
		      "conv_h": 4, 
		      "o_channels": 32, 
		      "pool_h": 2
		    },
		    {"conv_h": 4, "o_channels": 16, "pool_h": 2},
		    {"conv_h": 3, "o_channels": 8, "pool_h": 2},
		    {"conv_h": 3, "o_channels": 4, "pool_h": 2},
		    {"conv_h": 2, "o_channels": 2, "pool_h": 2},
		    {"conv_h": 2, "o_channels": 1, "pool_h": 2}
		  ],
		  "dense_conn_layers":
		  [
		    {"o_size": 1024}
		  ],
		  "out_size": 28
		}
		--------------------------------------------------------------------------------------

		Returns:
			True for success, False for otherwise
		"""
		self.fc_layers = []
		self.dc_layers = []
		self.softmax_layers = []
		if arch_dict:
			self.arch = arch_dict.copy()
		if not self.arch:
			return False
		# construct Frame Convolution Layers	
		in_channels = 1
		left_height = self.arch["in_height"]
		layer_id = 0
		for fcl_conf in self.arch["frame_conv_layers"]:
			w_shape = [fcl_conf["conv_h"], 1, in_channels, fcl_conf["o_channels"]]
			if "pool_h" in fcl_conf: 
				self.fc_layers.append(FrameConvLayer(w_shape, conv_h_stride = 1, layer_id = layer_id, pool_params = [fcl_conf["pool_h"], 2]))
				print("Layer %d, FramConvLayer | Conv HWIO,S = [%s],1; Pooling HW,S = [%d, 1],2" % 
						(layer_id, reduce(lambda x, y: str(x) + ',' + str(y), w_shape), fcl_conf["pool_h"]))
				left_height = (int(left_height / 2) + left_height % 2)
			else: 
				self.fc_layers.append(FrameConvLayer(w_shape, conv_h_stride = 1, layer_id = layer_id)) 
				print("Layer %d, FramConvLayer | Conv HWIO,S = [%s],1" % (layer_id, reduce(lambda x, y: str(x) + ',' + str(y), w_shape)))
			in_channels = fcl_conf["o_channels"]
			layer_id += 1
		# construct Densely Connect Layers
		in_size = left_height * self.arch["in_width"] * in_channels 
		for dcl_conf in self.arch["dense_conn_layers"]:
			w_shape = [in_size, dcl_conf["o_size"]]			
			self.dc_layers.append(DenseConnLayer(w_shape, layer_id = layer_id))
			in_size = dcl_conf["o_size"]
			print("Layer %d, DenseConnLayer | IO = [%s]" % (layer_id, reduce(lambda x, y: str(x) + ',' + str(y), w_shape)))
			layer_id += 1
		# construct Softmax Layer
		w_shape = [in_size, self.arch["out_size"]]
		self.softmax_layers.append(SoftmaxLayer(w_shape, layer_id = layer_id))
		print("Layer %d, SoftmaxLayer | IO = [%s]" % (layer_id, reduce(lambda x, y: str(x) + ',' + str(y), w_shape))) 
		layer_id += 1
		return True
	
	def propagate_to_classifier(self, x):
		"""Propagate to the last layer but don't activated by softmax.

		Args: 
			x: A 4-D tensor of shape [batch, height, width, channels] and type float32. The input of FCNN.

		Returns: 
			A 2-D tensor of shape [batch, out_size] and the same type of input. No-activated output of the last layer.
		"""
		z = x
		for fc_layer in self.fc_layers: 
			z = fc_layer.activate(z)		 
		for dc_layer in self.dc_layers:	
			z = dc_layer.activate(z)		 
		return self.softmax_layers[0].propagate_to(z)

	def classify(self, x):
		"""Feedforward from input to output, includes activation of softmax for classification. 

		Args:
			x: A 4-D tensor of shape [batch, height, width, channels] and type float32. The input of FCNN.

		Returns
			A 2-D tensor of shape [batch, out_size] and the same type of input. The output of the last layer.
		"""
		z = x
		for fc_layer in self.fc_layers: 
			z = fc_layer.activate(z)		 
		for dc_layer in self.dc_layers:	
			z = dc_layer.activate(z)		 
		return self.softmax_layers[0].activate(z)

	def feature_detect(self, x): 
		"""Feedforward from input to the last hidden layer. This operation could be used as feature detection.
		
		Args:
			x: A 4-D tensor of shape [batch, height, width, channels] and type float32. The input of FCNN.

		Returns
			A 2-D tensor of shape [batch, feat_size] and the same type of input. The feat_size equals the output_size of the last densely connect layer
		"""
		z = x
		for fc_layer in self.fc_layers: 
			z = fc_layer.activate(z)		 
		for dc_layer in self.dc_layers:	
			z = dc_layer.activate(z)		 
		return z

	def save_arch(self, out_file):
		"""Save FCNN architecture parameters in json format to file

		Args:
			out_file: A `string`. Path and name of output file

		Returns:
			True for success, False for otherwise.
		"""
		try:
			fp = open(out_file, 'w')
			if self.arch: 
				fp.write(json.dumps(self.arch) + '\n')
			return True
		except IOError as err:
			print("File Error: " + str(err))
			return False
		finally:
			fp.close()

	def read_arch(self, arch_file):
		"""Read FCNN architecture parameters in json format from file 
		
		Args:
			arch_file: A `string`. Path and name of input file

		Returns:
			True for success, False for otherwise.
		"""
		try: 
			fp = open(arch_file, 'r')
			arch = json.loads(fp.readline().rstrip())
			self.create(arch)
			return True
		except IOError as err:
			print("File Error: " + str(err))
			return False
		finally:
			fp.close()



