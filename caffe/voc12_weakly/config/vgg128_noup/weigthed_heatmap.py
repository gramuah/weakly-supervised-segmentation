#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random
HEATMAP_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_cam/heatmap_db/'

class WeigthedLabelLayer(caffe.Layer):


    def setup(self, bottom, top):
        
        # two tops: data and label
        if len(bottom) != 2:
            raise Exception("Do not define a bottom.")
        #self.iname = 0
        self.N = 20
        self.x = bottom[0].data
        self.y = bottom[1].data

    def reshape(self, bottom, top):     
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])


    def forward(self, bottom, top):
        
        for i in range(0, bottom[0].data.shape[0]):
            
            for c in range(0, self.N):
                #scipy.misc.toimage(bottom[0].data[i, c, ...], high=255, low=0).save(HEATMAP_FILE + '{}_{}.png'.format(self.iname, c))
                top[0].data[i, ...] = bottom[1].data[i, c] * bottom[0].data[i, c, ...]
            #self.iname +=1

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff[...] 
        for i in range(0, bottom[0].data.shape[0]):
            for c in range(0, self.N):
                bottom[1].diff[i,c] = np.mean(top[0].diff[i, c, ...])   
