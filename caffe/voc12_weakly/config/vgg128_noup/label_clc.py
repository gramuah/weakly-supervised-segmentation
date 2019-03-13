#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random

LABEL_PATH = '/home/carolina/projects/MESMERISE/DeepLab/benchmark_RELEASE/dataset/classlabels_ContextVOC12/'
BATCH_SIZE = 1
N_classes = 20

class PASCALContextLabelLayer(caffe.Layer):


    def setup(self, bottom, top):
        # config
        self.label_clc = np.zeros((BATCH_SIZE, N_classes))
	
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define 1 top: labels per images.")
        # data layers have no bottoms
        if len(bottom) != 2:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        
        self.indices = open('/home/carolina/projects/MESMERISE/DeepLab/benchmark_RELEASE/dataset/train_id.txt', 'r').read().splitlines()
        self.idx = 0

    def reshape(self, bottom, top):     
        top[1].reshape(BATCH_SIZE, N_classes)
	top[0].reshape(BATCH_SIZE, *(3, 224, 224))


    def forward(self, bottom, top):
        h = 0
        for i in range(0, BATCH_SIZE):
            self.label_clc[i, ...] = 0.0
            data_in = np.copy(bottom[1].data[i, ...])#caffe.io.load_image(curr_frame)
	    mask_in = np.copy(bottom[0].data[i, ...])#scipy.misc.imread(curr_mask) #cv2.imread(curr_mask, 0)#caffe.io.load_image(curr_mask)
	    data_in_res = np.zeros(shape=(3, 224,224), dtype=np.float32)
	    mask_in_res = np.zeros(shape=(1, 224,224), dtype=np.float32)
	    for jjj in range(0, 3):
	  	data_in_res[jjj, ...] = scipy.misc.imresize(data_in[jjj, ...], (224, 224), mode = 'F')
            
	    top[0].data[i, ...] = data_in_res
	    mask_in_res[0, ...] = scipy.misc.imresize(mask_in[0, ...], (224, 224), mode = 'F')
            mask_flatten = mask_in_res[0, ...].flatten()
	    #for db in range(0,len(mask_flatten)):
	    #	if mask_flatten[db]>0:
	    #		print mask_flatten[db]
	    clc_img = np.zeros(shape=(20), dtype=np.float32)
	    for cc in range(1,21):
		#clc_img[cc-1] -= 1
	  	idx2 = []
		idx2 = np.argwhere(mask_flatten == cc)
		if len(idx2) > 0:
			clc_img[cc-1] = 1
            # load image + label image pair
            #if (self.idx + i) > (len(self.indices)-1):
            #    l_aux = scipy.io.loadmat(LABEL_PATH + self.indices[h] + '.mat')
            #    h += 1
            #else:
            #    l_aux = scipy.io.loadmat(LABEL_PATH + self.indices[self.idx + i] + '.mat')
            
                        
            #lab = np.array(l_aux['label_clc']) 
            #plt.imshow(data_in_res.transpose(1, 2, 0).astype(np.uint8))
            #plt.show()
            #print clc_img
	    #raw_input()     
            self.label_clc[i, ... ] = clc_img
	

                  
            ##self.label = self.crop_label(self.label_res)
            # reshape tops to fit (leading 1 is for batch dimension)
        
        # assign output
        top[1].data[...] = self.label_clc

        
        # pick next input
        #if self.random:
        #    self.idx = random.randint(0, len(self.indices)-1)
        #else:
        self.idx += BATCH_SIZE
        if self.idx > (len(self.indices)-1):
            self.idx = h

    def backward(self, top, propagate_down, bottom):
        pass
