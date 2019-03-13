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

class PSLLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        if len(bottom) != 2:
            raise Exception("Need 1 input to compute the image label.")
        
    def reshape(self, bottom, top):
        N = 21
        ##self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(bottom[0].data.shape[0], N, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0

    def forward(self, bottom, top):
	for i in range(0, bottom[0].data.shape[0]):
		top[0].data[i, 0, ...] = bottom[0].data[i, 0, ...]
		for j in range(1,bottom[0].data.shape[1]):
			top[0].data[i, j, ...] = (bottom[1].data[i, j-1]) * (bottom[0].data[i, j, ...])
		
    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0
	bottom[1].diff[...] = 0.0
