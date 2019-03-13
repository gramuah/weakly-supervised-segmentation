#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/python')

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

class GTLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        if len(bottom) != 2:
            raise Exception("Need 1 input to compute the image label.")
        
    def reshape(self, bottom, top):
        N = 3
        ##self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(bottom[0].data.shape[0], N, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0

    def forward(self, bottom, top):
	for i in range(0, bottom[0].data.shape[0]):
		im = np.copy(bottom[0].data[i, ...])
		(ch,w,h) = im.shape
		im = im.reshape((1,w,h))
        	a = np.unique(im)
	       	a = a[(a < 255) & (a > 0)]	# gets all the pixels that are object classes
        	im3 = np.zeros((1,w,h))
		#im3[0,0, 0:len(a)] = a

		h = 0
		for j in range(0,20):
	        	if (bottom[1].data[i,j]):		
		        	im3[0,0, h] = j+1
				h += 1

		#im[im > 0] = 255
		
		im2 = np.zeros(shape=(1,41,41), dtype=np.float32)#np.copy(bottom[0].data[i, ...])
		
		
		max_val = 0.0
		db = 0.0
		for jj in range(0, 41):
			for ii in range(0, 41):
				if (im[0, jj, ii] == 255):
					max_val = 255 * np.max(bottom[1].data[i, ..., jj, ii])
					db += max_val
					
					im2[0, jj, ii]  = max_val
		#print np.sum(im2), db
		#raw_input()
		(ch,w2,h2) = im2.shape
		im2 = im2.reshape((1,w2,h2))


        	top[0].data[i, ...] = np.vstack((im,im2,im3))

    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0 #top[0].diff[...]
