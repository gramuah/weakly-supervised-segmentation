#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy
import random

PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'grid/'
SOURCEDIR = '/home/carolina/projects/MESMERISE/DeepLab/benchmark_RELEASE/dataset'

def BatchAdvancer(idx, buffer_size):
    idx += buffer_size
    return idx

class inputLayer(caffe.Layer):

  def setup(self, bottom, top):
      self.count = 0
      self.idx = 0
      self.N = 1
      self.buffer_size = 1
      self.channels = 3
      self.video_list = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid/list/train_aug.txt'
      self.height = 224
      self.width = 224
      
      f = open(self.video_list, 'r')
      f_lines = f.readlines()
      f.close()
      list_dict = {}
      self.video_order = []
      for ix, line in enumerate(f_lines):
          imgDir = line.split(' ')[0]
          maskDir = line.split(' ')[1]
          list_dict[ix] = {}
          list_dict[ix]['img'] = SOURCEDIR + imgDir
          list_dict[ix]['mask'] = SOURCEDIR + maskDir.strip()
          self.video_order.append(ix) 
      

      self.video_dict = list_dict
      self.num_videos = len(list_dict.keys())
      shape = (self.N, self.channels, self.height, self.width)
        
      self.transformer = caffe.io.Transformer({'data_in': shape})
      self.transformer.set_raw_scale('data_in', 255)
      image_mean = [104.008, 116.669, 122.675]
      channel_mean = np.zeros((3,224,224))
      for channel_index, mean_val in enumerate(image_mean):
        channel_mean[channel_index, ...] = mean_val
    
      self.transformer.set_mean('data_in', channel_mean)
      self.transformer.set_channel_swap('data_in', (2, 1, 0))
   
      self.transformer.set_transpose('data_in', (2, 0, 1))
        

  def reshape(self, bottom, top):
      top[0].reshape(self.N, *(self.channels, 224, 224))
      top[1].reshape(self.N, *(20,))
	
  def forward(self, bottom, top):
  
      data = []
      clc_vector = []      
      if self.idx + self.buffer_size >= self.num_videos:
          if (self.idx-self.num_videos < 0):
            idx_list = range(self.idx, self.num_videos)
            idx_list.extend(range(0, self.buffer_size-(self.num_videos-self.idx)))
          else:
            [dumm, idx_aux] = divmod(self.idx,self.num_videos)
            if idx_aux+self.buffer_size <= self.num_videos:
                idx_list = range(idx_aux, idx_aux+self.buffer_size)
            else:
                idx_list = range(idx_aux, self.num_videos)
                idx_list.extend(range(0, self.buffer_size-(self.num_videos-idx_aux)))
        
      else:
          idx_list = range(self.idx, self.idx+self.buffer_size)
          
      #print idx_list
      #(xmin xmax ymin ymax)
      grid_pos = [[0, 55, 0, 55], [56, 111, 0, 55], [112, 167, 0, 55], [168, 223, 0, 55], [0, 55, 56, 111], [56, 111, 56, 111], [112, 167, 56, 111], [168, 223, 56, 111], [0, 55, 112, 167], [56, 111, 112, 167], [112, 167, 112, 167], [168, 223, 112, 167], [0, 55, 168, 223], [56, 111, 168, 223], [112, 167, 168, 223], [168, 223, 168, 223]]
           
      for j in range(0, bottom[1].shape[0]):#idx_list:
          #key = self.video_order[j]
          #curr_frame = self.video_dict[key]['img']
	  #curr_mask = self.video_dict[key]['mask']
	  data_in_res = np.zeros(shape=(3, 224,224), dtype=np.float32)
	  mask_in_res = np.zeros(shape=(1, 224,224), dtype=np.float32)
          data_in = np.copy(bottom[1].data[j, ...])#caffe.io.load_image(curr_frame)
	  
	  mask_in = np.copy(bottom[0].data[j, ...])#scipy.misc.imread(curr_mask) #cv2.imread(curr_mask, 0)#caffe.io.load_image(curr_mask)
	  
	  for jjj in range(0, 3):
	  	data_in_res[jjj, ...] = scipy.misc.imresize(data_in[jjj, ...], (224, 224), mode = 'F')#caffe.io.resize_image(data_in, (224,224))#
	  
	  
	  mask_in_res[0, ...] = scipy.misc.imresize(mask_in[0, ...], (224, 224), mode = 'F')
	  #plt.imshow(data_in_res.transpose(1, 2, 0).astype(np.uint8))
          #plt.show()
          #processed_image = self.transformer.preprocess('data_in',data_in)
          
          data_aux = data_in_res #processed_image
	  mask_aux = mask_in_res
	  
          #TODO grid + random hidden position
          for pos in range(0,16):
            #p_idx = random.randint(0, 16)
            hid_p = random.randint(0,1)
            if hid_p:
                hidden_pos = grid_pos[pos]
                mask_aux[hidden_pos[0]:hidden_pos[1], hidden_pos[2]:hidden_pos[3]] = 255 
                data_aux[0, hidden_pos[0]:hidden_pos[1], hidden_pos[2]:hidden_pos[3]] = 122.675
                data_aux[1, hidden_pos[0]:hidden_pos[1], hidden_pos[2]:hidden_pos[3]] = 116.669
                data_aux[2, hidden_pos[0]:hidden_pos[1], hidden_pos[2]:hidden_pos[3]] = 104.008
          
          #plt.imshow(data_aux.transpose(1, 2, 0).astype(np.uint8))
          #plt.show()
          #scipy.misc.toimage(data_aux.transpose(1, 2, 0).astype(np.uint8), high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'map_clc_{}.png'.format(self.count))
          #self.count +=1
          an_flatten = mask_aux.flatten()
	  mask_flatten = mask_in_res[0, ...].flatten()
	  
	  #looking for classes
	  clc_img = np.zeros(shape=(20), dtype=np.float32)
	  for cc in range(1,21):
		#clc_img[cc-1] -= 1
	  	idx = []
		idx2 = []
	  	idx = np.argwhere(an_flatten == cc)
	  	idx2 = np.argwhere(mask_flatten == cc)
		
	  	if len(idx2) > 0:
			if len(idx) > 0:			
				clc_img[cc-1] = 1
	  
	 
	  
	  clc_vector.append(clc_img)
          data.append(data_aux)      
      
      self.idx = BatchAdvancer(self.idx, self.buffer_size)    
      for k in range(self.N): 
          top[0].data[k, ...] = data[k] 
	  top[1].data[k, ...] = clc_vector[k] 
	 
	  
    
  def backward(self, top, propagate_down, bottom):
      pass
