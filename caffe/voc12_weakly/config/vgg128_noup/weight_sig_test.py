#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random

BATCH_SIZE = 1

class WeightedLayer(caffe.Layer):


    def setup(self, bottom, top):
        # config
        self.repmat = np.zeros((BATCH_SIZE, 21, 65, 65))

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define 1 top: labels per images.")



        

    def reshape(self, bottom, top):     
        top[0].reshape(BATCH_SIZE, 21, 65, 65)
        top[0].data[...] = 0.0


    def forward(self, bottom, top):
        
        for i in range(0, BATCH_SIZE):
            top[0].data[i, 0, ... ] = bottom[0].data[i, 0, ...]	
	    #print bottom[1].data[i, ...]
	    #raw_input()		
            for cl in range(1, 21):
				

		if bottom[1].data[i, cl-1] < 0.1:
            		top[0].data[i, cl, ... ] = 0.0
		else:
            		top[0].data[i, cl, ... ] = bottom[0].data[i, cl, ...] #* bottom[1].data[i, cl]#np.multiply(bottom[0].data[i, cl, ...],bottom[1].data[i, cl, ...])
		#print self.repmat[i, cl, ... ]
		#raw_input()


                  
            ##self.label = self.crop_label(self.label_res)
            # reshape tops to fit (leading 1 is for batch dimension)
        
        



    def backward(self, top, propagate_down, bottom):
        #pass
        bottom[0].diff[...] = 0.0 
