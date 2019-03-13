#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
from numpy import unravel_index
import random

PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_cam/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'binary_mask/'


class SoftmaxLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.count = 0
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need 3 inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros(shape = (bottom[0].data.shape[0], 41*41, 21), dtype=np.float32)
        self.prob_local = np.zeros(shape = (bottom[0].data.shape[0], 41*41, 21), dtype=np.float32)
        self.y = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        self.y_mask_flatten = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        self.y_local = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
       
        #local region
        data_loss_local = 0.0
        for c in range(0, bottom[0].data.shape[0]):
            #img = np.zeros(shape=(bottom[0].data.shape[2],bottom[0].data.shape[3]), dtype=np.float32)
            #img = bottom[2].data[c, 0, ...]*255
            #scipy.misc.toimage(img, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + '{}.png'.format(self.count))
            self.count += 1
        
            scores = bottom[0].data[c, ...]
            y_mask = bottom[2].data[c, ...]
            self.y_mask_flatten[c, ...] = y_mask.flatten()
            self.y[c, ...] = bottom[1].data[c, ...].flatten()
            self.y[c, self.y[c, ...] == 255] = 0
            
            scores_1 = np.reshape(scores, (21, 41*41))
            
            scores_flatten = scores_1.T 
            
            exp_scores = np.exp(scores_flatten - np.max(scores_flatten, axis=1, keepdims=True))
            self.diff[c, ...] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
            
            
            for k in range(0, 41*41):
                if self.y_mask_flatten[c, k] > 0.0 : 
                    self.prob_local[c, k, ...] =  self.diff[c, k, ...]
                    self.y_local[c, k] = self.y[c, k]
                else: 
                    self.prob_local[c, k, 0] = 1.0 #self.probs[k, 0] #1.0  
       
            data_loss_local = -np.sum(np.log(self.prob_local[c, np.arange(self.prob_local.shape[1]), np.array(self.y_local[c, ...],dtype=np.uint16)]))/ float(41*41)  
            
            #/ N-np.sum(np.log(aux3[np.arange(N), np.array(aux,dtype=np.uint16)])) #/ N
            #correct_logprobs = -np.log(probs[0, range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16)])
            #data_loss = np.sum(correct_logprobs)#/bottom[0].num

        top[0].data[...] = np.mean(data_loss_local) #+ data_loss_local

        
    def backward(self, top, propagate_down, bottom):
        
        
        for i in range(1):
            for c in range(0, bottom[0].data.shape[0]):
                delta_local_flatten = self.prob_local[c, ...]
                if not propagate_down[i]:
                    continue
                if i == 0:
                
                    delta_local_flatten[np.arange(self.prob_local.shape[1]), np.array(self.y_local[c, ...],dtype=np.uint16)] -= 1
            
            
                delta_local_res =  delta_local_flatten.T
            
            
                delta_local = np.reshape(delta_local_res, (21, 41, 41))
           
                bottom[0].diff[c, ...] = delta_local/float(41*41)
                
        bottom[1].diff[...] = 0.0                
        bottom[2].diff[...] = 0.0
