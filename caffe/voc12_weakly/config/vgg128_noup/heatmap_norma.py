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
import cv2
from numpy import unravel_index

PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_cam_v1/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'mask/'
class NormLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        if len(bottom) != 1:
            raise Exception("Need 1 input to compute the image label.")
        
    def reshape(self, bottom, top):
        N = 20 #classes + back
        ##self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(bottom[0].data.shape[0], N, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0
       
        
    
    def forward(self, bottom, top):
        max_value = -1000000000000000000000000000
        min_value = 1000000000000000000000000000
        for i in range(0,bottom[0].data.shape[0]):
            self.count +=1
            print self.count
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
            
            
            thres = 0.2
            
            for j in range(0,bottom[0].data.shape[1]):
                top[0].data[i, j, ...] = (bottom[0].data[i, j, ...] - min_value)/float(max_value - min_value)

            
    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0 #top[0].diff[...]
