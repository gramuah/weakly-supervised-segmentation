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


class CondSoftmaxLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        self.epoch = 0
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros(shape = (bottom[0].data.shape[0], 41*41, 21), dtype=np.float32)
	self.scores_1 = np.zeros(shape = (bottom[2].data.shape[0], 21, 41*41), dtype=np.float32)
        self.y = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
	self.cond = np.zeros(shape = (bottom[0].data.shape[0], 1), dtype=np.float32)
	self.L_plus = np.zeros(shape = (bottom[0].data.shape[0], 1), dtype=np.float32)
	self.L_minus = np.zeros(shape = (bottom[0].data.shape[0], 1), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        data_loss = 0.0
        for c in range(0, bottom[0].data.shape[0]):
            scores = np.copy(bottom[2].data[c, ...])
	    
	    mask = np.copy(bottom[0].data[c, ...])
	    idx = np.argwhere(mask > 0)
	    self.cond[c, 0] = len(idx) #np.sum(mask)
	    
	    self.y[c, ...] =mask.flatten()
            
	    self.scores_1[c, ...] = np.reshape(scores, (21, 41*41))
	    
	    scores_flatten = self.scores_1[c, ...].T
            
	    exp_scores = np.exp(scores_flatten - np.max(scores_flatten, axis=1, keepdims=True))
	    self.diff[c, ...] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
	 
	    probs = self.diff[c, ...]

	    if (self.cond[c, 0] > 0):
                #print self.diff[c, np.arange(self.diff.shape[1]), np.array(self.y[c, ...] ,dtype=np.uint16)]
           	correct_logprobs = -np.log(self.diff[c, np.arange(self.diff.shape[1]), np.array(self.y[c, ...] ,dtype=np.uint16)])
            	data_loss += np.sum(correct_logprobs)/float(41*41)
	    
	    else:	
		self.L_plus[c, 0] = 1 #background class
		for ac in range(0, bottom[1].data.shape[1]):
			if (bottom[1].data[c, ac]):
				self.L_plus[c, 0] += 1	
		self.L_minus[c, 0] = 21 - self.L_plus[c, 0] 
           	for cc in range(0, bottom[0].data.shape[1]):
			#Sums over classes in image (or background) where s is max-scoring pixel for the class
			if ((cc == 0) or (bottom[1].data[c, cc-1])): 
				position_max = np.argmax(self.scores_1[c, cc, ...])
				Sc = probs[position_max, cc]
				data_loss -= (np.log(Sc)/float(self.L_plus[c, 0]));
					
			else:
				position_max = np.argmax(self.scores_1[c, cc, ...])
				Sc = probs[position_max, cc]
				if(Sc == 1.0):
					data_loss -= (np.log(0.001)/float(self.L_minus[c, 0]));
				else:					
					data_loss -= (np.log(1-Sc)/float(self.L_minus[c, 0]));
					
			
            
        
        top[0].data[...] = data_loss / bottom[2].num

    def backward(self, top, propagate_down, bottom):
        
        #delta_flatten = []
        for i in range(3):
            for c in range(0, bottom[0].data.shape[0]):
                
                if not propagate_down[i]:
                    continue

		
		if (self.cond[c, 0] > 0):
		    delta_flatten = np.copy(self.diff[c, ...])
                    delta_flatten[np.arange(self.diff.shape[1],dtype=np.uint16), np.array(self.y[c, ...],dtype=np.uint16)] -= 1  

		    #bottom[0].diff[c, ...] /= float(21)

		else:
		    delta_flatten = np.zeros(shape = (41*41, 21), dtype=np.float32)
		    for cc in range(0, bottom[2].data.shape[1]):
			if ((cc == 0) or (bottom[1].data[c, cc-1])): 
				position_max = np.argmax(self.scores_1[c, cc, ...])
				delta_flatten[position_max, cc] = ((self.diff[c, position_max, cc] - 1)/float(self.L_plus[c, 0]))
    				for aux in range(0, bottom[2].data.shape[1]):
					if aux != cc:
						delta_flatten[position_max, aux] = (self.diff[c, position_max, aux]/float(self.L_plus[c, 0]))
	
			else:
				position_max = np.argmax(self.scores_1[c, cc, ...])
				Sc = self.diff[c, position_max, cc]
				term = Sc #-Sc + (Sc/(1-Sc))
				delta_flatten[position_max, cc] += (term/float(self.L_minus[c, 0]));
				
    				for aux in range(0, bottom[2].data.shape[1]):
					if aux != cc:
						Sc_aux = self.diff[c, position_max, aux]
						if Sc == 1:
							tr1 = 1.0
						else:
							tr1 = (Sc_aux*Sc)/float(1-Sc)
						delta_flatten[position_max, aux] -= (tr1/float(self.L_minus[c, 0]));
						
						#print delta_flatten[position_max, aux]


		delta_res = delta_flatten.T
                delta = np.reshape(delta_res, (21, 41, 41))
		if (self.cond[c, 0] > 0):
			bottom[2].diff[c, ...] = delta/float(41*41)
		else:
                    	bottom[2].diff[c, ...] = delta
        
        bottom[2].diff[...] /= bottom[2].num #python euclidean loss function
        bottom[1].diff[...] = 0.0
        bottom[0].diff[...] = 0.0
