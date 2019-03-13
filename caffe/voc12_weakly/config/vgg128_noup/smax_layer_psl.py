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
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_woback/gt/'
PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_cam/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'mask/'

class SoftmaxLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        self.epoch = 0
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros(shape = (bottom[0].data.shape[0], 41*41, 21), dtype=np.float32)
        
        self.psl_mask = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        data_loss = 0.0
        
        for c in range(0, bottom[0].data.shape[0]):
            scores = bottom[0].data[c, ...]
            ##PSL
            weig_scores = bottom[0].data[c, ...] 
            for cl in range(1,21):
                w = bottom[1].data[c, cl-1]
                weig_scores[cl, ...] = weig_scores[cl, ...]*w
            self.psl_mask = np.argmax(weig_scores, axis=0).flatten()
        
            
            scores_1 = np.reshape(scores, (21, 41*41))
            scores_flatten = scores_1.T 
            
            exp_scores = np.exp(scores_flatten - np.max(scores_flatten, axis=1, keepdims=True))
            self.diff[c, ...] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
            probs = self.diff[c, ...]
            
            
            correct_logprobs = -np.log(self.diff[c, np.arange(self.diff.shape[1]), np.array(self.psl_mask[c, ...] ,dtype=np.uint16)])
            data_loss += np.sum(correct_logprobs)/float(41*41)
            
            
            
        
        top[0].data[...] = (data_loss / float(bottom[0].data.shape[0])) 
    def backward(self, top, propagate_down, bottom):
        
        #delta_flatten = []
        for i in range(1):
            for c in range(0, bottom[0].data.shape[0]):
                
                delta_flatten_psl = self.diff[c, ...]
                if not propagate_down[i]:
                    continue
                if i == 0:
                    delta_flatten_psl[np.arange(self.diff.shape[1],dtype=np.uint16), np.array(self.psl_mask[c, ...],dtype=np.uint16)] -= 1
                    delta_res_psl = delta_flatten_psl.T
                    delta_psl = np.reshape(delta_res_psl, (21, 41, 41))
                    
                    bottom[0].diff[c, ...] = delta_psl/float(41*41)
                    #smx standar             
                    #delta_flatten[np.arange(self.diff.shape[1],dtype=np.uint16), np.array(self.y[c, ...],dtype=np.uint16)] -= 1 #[] # 
                    #delta_res = delta_flatten.T
                    
                    #delta = np.reshape(delta_res, (21, 41, 41))
  
                    #bottom[0].diff[c, ...] = delta/float(41*41)
                    
        
        bottom[0].diff[...] /= float(bottom[0].data.shape[0]) 
        
        bottom[1].diff[...] = 0.0
