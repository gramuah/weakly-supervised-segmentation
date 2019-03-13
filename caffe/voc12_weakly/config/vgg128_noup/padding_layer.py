#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import unravel_index
import random
PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_cam_255/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'padd/'

class PaddingLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        if len(bottom) != 1:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], *(1, 41, 41))

    def forward(self, bottom, top):
        for c in range(0, bottom[0].data.shape[0]):
            pixel_padd = np.copy(bottom[0].data[c, ...])
            #print pixel_padd.shape	            
            for ii in range(1, bottom[0].data.shape[2]-1):
                for jj in range(1, bottom[0].data.shape[3]-1):
                    if pixel_padd[0, ii,jj] == 0: #bg pixel
                        #print "hola "
                        #print pixel_padd[ii-2, jj]
                        #print pixel_padd[ii-1, jj]
                        #print pixel_padd[ii+1, jj]
                        #raw_input()
                        if (pixel_padd[0,ii-1, jj] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii+1, jj] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii, jj - 1] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii, jj + 1] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii-1, jj -1] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii+1, jj -1] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii -1, jj + 1] != 0 and pixel_padd[0,ii-1, jj] != 255) or (pixel_padd[0, ii+1, jj +1] != 0 and pixel_padd[0,ii-1, jj] != 255):
                           pixel_padd[0, ii,jj] = 255 
            
            
            
            top[0].data[c, ...] = pixel_padd
            
            
    def backward(self, top, propagate_down, bottom):
	bottom[0].diff[...]=0.0
