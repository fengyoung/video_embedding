# Video-Embedding

This is Implement of Video Embedding based on ***Tensorflow, Inception-V3 & FCNN(Frames Supported Convolution Neural Network)*** 

Video Embeding can be interpreted as the processing of **Video Features Extraction**. As figure-1, a video consisits of a lots of images (named *frame*). Each frame contants image concept, and these concept can be organized to a matrix in time sequence (we call it **Video-Matrix**) which can be used for representing the content of the video in frame leval details. The extraction of Video-Matrix is as the same as image embedding. We choose Inception-v3 to solve it.

However, we don't need the frame details in some video applications, for example, Video-Retrieval, Recommendation etc. So we prepose a new CNN to compress the 2-D Video-Matrix to a 1-D **Video-Vector**. 

<img src="https://raw.githubusercontent.com/fengyoung/video_embedding/master/pic/video_embedding_01.jpeg" alt="figure-1. The Processing of Video Embedding" align=center />

***figure-1. The Processing of Video Embedding***

The new CNN, which is called **FCNN (Frames Supported Convolution Neural Network)**, contains 6 *f_conv* layers, 1 *densely connect* layer and 1 *softmax* layer. The output of densely connect layer (the last layer before softmax) is the Video-Vector. FCNN is trained by using Weibo-MCN classificated data-set, and it also can be used for video classification.

<img src="https://raw.githubusercontent.com/fengyoung/video_embedding/master/pic/fcnn_arch.jpg" width = "450" alt="figure-2. The architecture of FCNN" align=center />

***figure-2. The architecture of FCNN***

## 1. Requirements

- Python 3.x
- Tensorflow >= 1.0
- OpenCV >= 3.2

## 2. Installation

**(1) Install TensorFlow**

See [Installing TensorFlow](https://www.tensorflow.org/install/) for instructions on how to install the release binaries or how to build from source.

**(2) Install OpenCV on Python**

You can find the introduction and Download link at [OpenCV-3.2](http://opencv.org/opencv-3-2.html) or other websites. Choose the right version (over version 3.2) to install on your enverimant. 

**(2) Clone the source of video_embedding**

```
git clone https://github.com/fengyoung/video_embedding.git <YOUR REPO PATH>
```

## 3. How to Use

### 3.1 Video Features Extraction
There are 2 levels of video features: **Video-Matrix** & **Video-Vector**,  

### 3.2 Video Classification

### 3.3 FCNN Training









```
./config	                       		 // config examples
./data/label_14k                         // label mapping of sample data set
./data/wb_14k_h30w2048_pattern_v2_part   // sample data in string proto
./data/wb_14k_h30w2048_tfrecord_v2_part  // sample data in tfrecord proto
./model/mcn_14k_c30_fcnn_model           // a completed FCNN model based on 14,000 weibo MCN videos
./video_embedding/comm                   // comm package  source code in python 
./video_embedding/image_embedding		 // implement of image embedding
./video_embedding/framdes_embedding		 // implement of frames embedding
./video_embedding/tools                  // some useful tools for label collection, format converting, ...
```

## 4. How to Use

### 4.1 Configure

You can find the example of configure in ***./config***

The proto of configure is as following: 

```
{
	"fcnn_arch":	# FCNN architecture parameters 
	{
		"in_height": 30,	# height of input video matrix  
		"in_width": 2048,	# width of input video matrix  
		"frame_conv_layers":	# Frame Convolutional Layers 
		[
			{ 
				"conv_h": 4,		# height of conv core 
				"o_channels": 32,	# number of channels of current layer
				"pool_h": 2			# max-pooling height
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
```


### 4.2 Training a FCNN model

You can excute following shells to train an FCNN model for example. 

```
cd ./video_embedding/
python3.4 fcnn_train.py ../../config/video2vec_fcnn_config.json ../../data/wb_14k_h30w2048_tfrecord_v2_part/ ./out_model 
```

The model would be trained from video-matrix pattern data in *"../../data/wb_14k_h30w2048_tfrecord_v2_part/"*, and output to *"./out_model"*.

*"../../config/video2vec_fcnn_config.json"* is the configure file. 



### 4.3 Classification

You can excute following shells to classify each sample from pattern files. 

```
cd ./video_embedding/
python3.4 fcnn_pred.py ../../model/mcn_14k_c30_fcnn_model/ ../../data/wb_14k_h30w2048_pattern_v2_part pattern
```

*"../../model/mcn_14k_c30_fcnn_model/"* is a prepared FCNN model. 

*"../../data/wb_14k_h30w2048_pattern_v2_part"* is the path of testing samples. 

*"pattern"* indicates the suffix of file in samples path


### 4.4 Video-Level Feature Extraction (Video2Vec)

You can excute following shells to extract the Video-Vec from pattern files. 

```
cd ./video_embedding/
python3.4 video2vec.py ../../model/mcn_14k_c30_fcnn_model/ ../../data/wb_14k_h30w2048_pattern_v2_part pattern ./video_vec.out
```

*"../../model/mcn_14k_c30_fcnn_model/"* is a prepared FCNN model. 

*"../../data/wb_14k_h30w2048_pattern_v2_part"* is the path of testing samples

*"pattern"* indicates the suffix of file in samples path

*"./video_vec.out"* is the out file 


## 5. Important

There are two supported types of the Video-Matrix Patterns (VMP): pattern-string & tfrecord-example 

### 5.1 Pattern-String proto (v2)

```
mid,labelid0_labelid1_labelid2,height_width,x0_x1_x2_..._xn
mid,labelid2_labelid5,height_width,x0_x1_x2_..._xn
...
```

### 5.2 Tfrecord-Example proto (v2)


```
features: {
	feature: {
		key: "mid"
		value: {
			bytes_list: {
				value: [mid string]
			}
		}
	}
	feature: {
		key: "off"
		value: {
			int64_list: {
				value: [segment offset] 
			}
		}
	}
	feature: {
		key: "label"
		value: {
			bytes_list: {
				value: ["0,3,7"]    # labelid list string 
			}
		}
	}
	feature: {
		key: "size"
		value: {
			int64_list: {
				value: [v_height, v_width]
			}
		}
	}
	feature: {
		key: "feature"
		value: {
			float_list: {
			value: [(v_height * width) float features]
			}
		}
	}
}
```

