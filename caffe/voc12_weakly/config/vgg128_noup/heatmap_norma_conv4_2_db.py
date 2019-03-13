#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.io as sio
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

from numpy import unravel_index

IMAGE_FILE_MAT_PROB = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/prob_mat/'
IMAGE_FILE_MAT = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/heat_mat/'
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/gt_heat/'
PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'mask_new/'
IMAGE_FILE_BMMASK_FCN_1 = PATH_RES + 'conv5_new/'
IMAGE_FILE_BMMASK_FCN_2 = PATH_RES + 'conv5fusion/'

class NormLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        self.epoch = 0
        if len(bottom) != 6:
            raise Exception("Need 6 input to compute the image label.")
        
    def reshape(self, bottom, top):
        N = 21 #classes + back
        ##self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(bottom[0].data.shape[0], N, bottom[0].data.shape[2], bottom[0].data.shape[3])
	top[1].reshape(bottom[0].data.shape[0], 1, 1, 2)        
	top[0].data[...] = 0.0
       	
        
    
    def forward(self, bottom, top):

        #db
        background = [0,0,0]
        aeroplane = [128,128,128]
        bicycle = [128,0,0]
        bird = [192,192,128]
        boat = [255,69,0]
        bottle = [128,64,128]
        bus = [60,40,222]
        car = [128,128,0]
        cat = [192,128,128]
        chair = [64,64,128]
        cow = [64,0,128]
        diningtable = [64,64,0]
        dog = [0,128,192]
        horse = [32,40,0]
        motorbike = [67, 123, 222]
        person = [134, 2, 223]
        pottedplant = [22, 128, 233]
        sheep = [100, 2, 2]
        sofa = [56, 245, 32]
        train =[200, 100, 10]
        tvmonitor = [99, 89, 89]

        
        for i in range(0,bottom[0].data.shape[0]):
            top[0].data[i, 0, ...] = 1.0
	    top[1].data[i, 0, 0, ...] = 41
            aux_c = np.mean(bottom[1].data[i, ...], axis = 0)
            #aux_save_1 = scipy.misc.imresize(aux_c, (100,100))
            #scipy.misc.toimage(aux_save_1, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_2 + 'map_clc_{}.png'.format(self.count))

            aux_cc = scipy.misc.imresize(np.mean(bottom[2].data[i, ...], axis = 0), (14,14))
            #aux_save_2 = scipy.misc.imresize(aux_cc, (100,100))
            #scipy.misc.toimage(aux_save_2, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_1 + 'map_clc_{}.png'.format(self.count))
            aux_c5 = np.mean(bottom[3].data[i, ...], axis = 0)
            
            aux_ccc = (aux_c + aux_cc + aux_c5)/3.0 #aux_c + 
            aux_ccc= (aux_ccc -np.min(aux_ccc))/float(np.max(aux_ccc)-np.min(aux_ccc))
            
            #aux2 = np.zeros(shape=(14*14), dtype=np.float32)
            #aux3 = np.zeros(shape=(14,14), dtype=np.float32)
            #auxi = aux_ccc.flatten()
            #idx = []
            #idx = np.argwhere(auxi < 0.3*np.max(aux_ccc))

            #aux2[idx] = 1.0
            #aux3 = np.reshape(aux2, (14,14))
            top[0].data[i, 0, ...] -= aux_ccc
            

            max_value = -1000000000000000000000000000
            min_value = 1000000000000000000000000000
            aux = np.zeros(shape=(14,14), dtype=np.float32)
            aux2 = np.zeros(shape=(14,14), dtype=np.float32)
            aux3 = np.zeros(shape=(14*14), dtype=np.float32)
            aux4 = np.zeros(shape=(14*14), dtype=np.float32)
            db = 0
            for j in range(0,bottom[0].data.shape[1]):
                heat_map_flatten =bottom[0].data[i, j, ...].flatten()
                max_value_aux = np.max(heat_map_flatten)
                min_value_aux = np.min(heat_map_flatten)
                if (max_value_aux > max_value):
                    max_value = max_value_aux
                    db = j
                if (min_value_aux < min_value):
                    min_value = min_value_aux
  
           

            thres = 0.2*max_value
            
            for j in range(0,bottom[0].data.shape[1]):
				if bottom[5].data[i, j] == 0.0: #(np.max(bottom[0].data[i, j, ...]) < 0.0):
					top[0].data[i, j+1, ...] = 0.0
				else:
					top[0].data[i, j+1, ...] = (bottom[0].data[i, j, ...] - np.min(bottom[0].data[i, j, ...]))/float(np.max(bottom[0].data[i, j, ...]) - np.min(bottom[0].data[i, j, ...]))#bottom[4].data[i,j]*((bottom[0].data[i, j, ...] - min_value)/float(max_value - min_value))

				
                
            
            
    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0 #top[0].diff[...]
