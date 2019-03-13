#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random

LABEL_PATH = '/home/carolina/Downloads/Felzenszwalb-Segmentation-master/pascal_test_513/'
BATCH_SIZE = 1

class SPinputLayer(caffe.Layer):


    def setup(self, bottom, top):
               
	
        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define 1 top: labels per images.")
        # data layers have no bottoms
        
        # load indices for images and labels
        
        self.indices = open('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/prueba/list/val_id.txt', 'r').read().splitlines()
        self.idx = 0

    def reshape(self, bottom, top):     
        
	top[0].reshape(BATCH_SIZE, *(1, 513, 513))


    def forward(self, bottom, top):
        h = 0
        for i in range(0, BATCH_SIZE):
            if (self.idx + i) > (len(self.indices)-1):
                mask_in = scipy.misc.imread(LABEL_PATH + self.indices[h] + '.png')
                h += 1
            else:
                mask_in = scipy.misc.imread(LABEL_PATH + self.indices[self.idx + i] + '.png')
            
	    
	    top[0].data[i, ...] = mask_in


	self.idx += BATCH_SIZE
        if self.idx > (len(self.indices)-1):
            self.idx = h
