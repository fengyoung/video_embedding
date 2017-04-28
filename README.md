# Video-Embedding

This is Video Embedding based on Tensorflow & FCNN(Frames Supported Convolution Neural Network) 


## 1. List

./config	                             // config examples

./data/label_14k                         // label mapping of sample data set

./data/wb_14k_h30w2048_pattern_v2_part   // sample data in string proto

./data/wb_14k_h30w2048_tfrecord_v2_part  // sample data in tfrecord proto

./model/mcn_14k_c30_fcnn_model           // a completed FCNN model based on 14,000 weibo MCN videos

./video_embedding/comm                   // comm package  source code in python 

./video_embedding/framdes_embedding		 // implement of frames embedding

./video_embedding/tools                  // some useful tools for label collection, format converting, ...



## 2. How to Use

### 2.1 Configure

You can find the example of configure in ./config

The proto of configure is as following: 

---------------------------------------------------------
{
	"fcnn_arch":	# FCNN architecture parameters 
	{
		"in_height": 30,	# height of input video matrix  
		"in_width": 2048,	# width of input video matrix  
		"frame_conv_layers":	# Frame Convolutional Layers 
		[
			{ 
			  "conv_h": 4,	# height of conv core 
			  "o_channels": 32, # number of channels of current layer
			  "pool_h": 2		# max-pooling height
			},
			{"conv_h": 4, "o_channels": 16, "pool_h": 2},
			{"conv_h": 3, "o_channels": 8, "pool_h": 2},
			{"conv_h": 3, "o_channels": 4, "pool_h": 2},
			{"conv_h": 2, "o_channels": 2, "pool_h": 2},
			{"conv_h": 2, "o_channels": 1, "pool_h": 2}
		],
		"dense_conn_layers":
		[
			{"o_size": 1024} # output size of densely connect layer
		],
		"out_size": 28		# out size of FCNN
	},
	"train_params":	# training parameters
	{
		"max_epochs": 100, 
		"early_stop": -1,
		"batch_size": 100,
		"shuffle": false,
		"epsilon": 0.1 
	}
}
---------------------------------------------------------



## 3. Important



