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
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/gt_todas/'
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
        self.y = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        self.psl_mask = np.zeros(shape = (bottom[0].data.shape[0], 41*41), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        data_loss = 0.0
        
        for c in range(0, bottom[0].data.shape[0]):
            scores = bottom[0].data[c, ...]
            
            ##scores ared clustered into FG/BG
            #bg_mask = np.zeros(shape = (1, 41, 41), dtype=np.float32)
            #dl_mask = np.argmax(bottom[0].data[c, ...], axis=0)
            #bg_mask[0, dl_mask == 0] = 1
            ##concatenate with the CAM
            #score_map_aug = np.concatenate((bg_mask, bottom[1].data[c, ...]), axis=0)
            #aaa = np.argmax(bottom[1].data[c, ...], axis=0)
            #scipy.misc.toimage(aaa, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'mask_{}.png'.format(self.count))
            #self.count += 1
            
            
            ##Plot the mask
            ind = np.argmax(bottom[0].data[c, ...], axis=0)
            r = ind.copy()
            g = ind.copy()
            b = ind.copy()
            
            background = [0,0,0]
            aeroplane = [128,128,128]
            bicycle = [128,0,0]
            bird = [192,192,128]
            boat = [255,69,0]
            bottle = [128,64,128]
            bus = [60,40,222]
            car = [128,128,0]
            cat = [192,128,128]
            chair = [64,64,128]
            cow = [64,0,128]
            diningtable = [64,64,0]
            dog = [0,128,192]
            horse = [32,40,0]
            motorbike = [67, 123, 222]
            person = [134, 2, 223]
            pottedplant = [22, 128, 233]
            sheep = [100, 2, 2]
            sofa = [56, 245, 32]
            train =[200, 100, 10]
            tvmonitor = [99, 89, 89]


            label_colours = np.array([background, aeroplane,  bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor])
            for l in range(0,21):
		        r[ind==l] = label_colours[l,0]
		        g[ind==l] = label_colours[l,1]
		        b[ind==l] = label_colours[l,2]
		        
            rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
            rgb[:,:,0] = r/255.0
            rgb[:,:,1] = g/255.0
            rgb[:,:,2] = b/255.0
    
            #if (self.count == 0) or (self.count == 10582) or (self.count == 21164) or (self.count == 31746) or (self.count == 42328) or (self.count == 52910) or (self.count == 63492) or (self.count == 74074) or (self.count == 84656) or (self.count == 95238):
	    #        aux_save = scipy.misc.imresize(rgb, (200,200)) 
	    scipy.misc.toimage(rgb, high=255, low=0).save(IMAGE_FILE + 'img_{}.png'.format(self.count))
	    #        self.epoch += 1
            
            self.count += 1
            self.y[c, ...] = bottom[1].data[c, ...].flatten() #np.argmax(bottom[1].data[c, ...], axis=0).flatten()# np.argmax(score_map_aug, axis=0).flatten() #
            #self.y[c, self.y[c, ...] == 255] = 0
            
            scores_1 = np.reshape(scores, (21, 41*41))
            scores_flatten = scores_1.T 
            
            exp_scores = np.exp(scores_flatten - np.max(scores_flatten, axis=1, keepdims=True))
            self.diff[c, ...] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
            probs = self.diff[c, ...]
            
            
            correct_logprobs = -np.log(self.diff[c, np.arange(self.diff.shape[1]), np.array(self.y[c, ...] ,dtype=np.uint16)])
            data_loss += np.sum(correct_logprobs)/float(41*41)
            
            
            
        
        top[0].data[...] = data_loss / float(bottom[0].data.shape[0])

    def backward(self, top, propagate_down, bottom):
        
        #delta_flatten = []
        for i in range(1):
            for c in range(0, bottom[0].data.shape[0]):
                
                delta_flatten = self.diff[c, ...]
                delta_flatten_psl = self.diff[c, ...]
                if not propagate_down[i]:
                    continue
                if i == 0:
                    delta_flatten[np.arange(self.diff.shape[1],dtype=np.uint16), np.array(self.y[c, ...],dtype=np.uint16)] -= 1
                    delta_res = delta_flatten.T
                    delta = np.reshape(delta_res, (21, 41, 41))
                    
                    bottom[0].diff[c, ...] = delta/float(41*41)
                    #smx standar             
                    #delta_flatten[np.arange(self.diff.shape[1],dtype=np.uint16), np.array(self.y[c, ...],dtype=np.uint16)] -= 1 #[] # 
                    #delta_res = delta_flatten.T
                    
                    #delta = np.reshape(delta_res, (21, 41, 41))
  
                    #bottom[0].diff[c, ...] = delta/float(41*41)
                    
        
        bottom[0].diff[...] /= float(bottom[0].data.shape[0]) 
        
        bottom[1].diff[...] = 0.0
