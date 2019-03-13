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


class SoftmaxLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        #check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        
        # loss output is scalar
        top[0].reshape(1)
        

    def forward(self, bottom, top):
        self.probs = np.zeros(shape = (bottom[0].data.shape[0], 41*41, 21), dtype=np.float32)
        self.y_flatten= np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        self.Nh = np.zeros(shape = (1,bottom[0].data.shape[0]), dtype=np.float32)
        self.Nw = np.zeros(shape = (1,bottom[0].data.shape[0]), dtype=np.float32)
        self.N = np.zeros(shape = (1,bottom[0].data.shape[0]), dtype=np.float32)
        #TODO batch size
        data_loss = 0
        for i in range(0, bottom[0].data.shape[0]):
            
            self.Nh[0,i] =  bottom[0].data[i].shape[1]
            self.Nw[0,i] =  bottom[0].data[i].shape[2] 
            self.N[0,i] = self.Nh[0,i]*self.Nw[0,i]
            
            
            scores = bottom[0].data[i, ...]
            
            y = bottom[1].data[i, ...]
            
            #y = np.argmax(bottom[1].data[i, ...], axis=0) 
            #scores = np.squeeze(scores)
            #y = np.squeeze(y)
            
            #self.y_flatten[i, ...] = np.argmax(bottom[1].data[i, ...], axis=0).flatten()
            self.y_flatten[i, ...] = bottom[1].data[i, ...].flatten()
            self.y_flatten[i, self.y_flatten[i, ...] == 255] = 0
            
           
            
            scores_1 = np.reshape(scores, (21, self.N[0,i]))
            scores_flatten = scores_1.T 
           
            

            self.probs[i, ...] = np.exp(scores_flatten - np.max(scores_flatten, axis=1, keepdims=True))
           
            
            self.probs[i, ...] /= np.sum(self.probs[i, ...], axis=1, keepdims=True)

            data_loss = -np.sum(np.log(self.probs[i, np.arange(self.N[0,i],dtype=np.uint16), np.array(self.y_flatten[i, ...],dtype=np.uint16)]))/float(self.N[0,i])
            
            
            #/ N-np.sum(np.log(aux3[np.arange(N), np.array(aux,dtype=np.uint16)])) #/ N
            #correct_logprobs = -np.log(probs[0, range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16)])
            #data_loss = np.sum(correct_logprobs)#/bottom[0].num

        top[0].data[...] = np.mean(data_loss)

        
    def backward(self, top, propagate_down, bottom):

        
        for i in range(1):
            for c in range(0,bottom[0].data.shape[0]):
                delta_flatten = self.probs[c, ...]
            #if not propagate_down[i]:
            #    continue
                if i == 0:

                    delta_flatten[np.arange(self.N[0,c],dtype=np.uint16), np.array(self.y_flatten[c, ...],dtype=np.uint16)] -= 1

                    delta_res = delta_flatten.T
                    
                    delta = np.reshape(delta_res, (21, self.Nh[0,c], self.Nw[0,c]))
                    bottom[i].diff[c, ...] = delta/float(self.N[0,c])
            
                    
                    bottom[1].diff[c, ...] = 0.0
