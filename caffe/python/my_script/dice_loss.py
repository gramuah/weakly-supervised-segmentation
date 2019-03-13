#!/usr/bin/env python
import sys
sys.path.append('../../../../python')
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


class DiceLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need 2 inputs to compute the DICE distance.")
            
    def reshape(self, bottom, top):
        # check input dimensions match
        #if (bottom[0].count != bottom[1].count): 
        #    print bottom[0].count
        #    print bottom[1].count
            
            #raise Exception("Inputs must have the same dimension.")
        
        # difference is shape of inputs

        self.diff1 = np.zeros_like(bottom[1].data, dtype=np.float32)
        
        # loss output is scalar
        top[0].reshape(1)
        
    def forward(self, bottom, top):
        dice = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        
        for i in range(0,bottom[1].data.shape[0]):

            ind =  np.argmax(bottom[1].data[i, ...], axis=0) #(bottom[1].data) #Predict

           
            y_true_aux = np.array(bottom[0].data[i, ...]) #GT mask
            y_pred_aux = np.array(ind)
        
        
        
            y_true_f = y_true_aux.flatten()
            y_pred_f = y_pred_aux.flatten()
            
            #print i
            #print y_pred_f
            #raw_input()
        
            diff_aux_0 =np.zeros_like(y_pred_f, dtype=np.float32)
            diff_aux_1 =np.zeros_like(y_pred_f, dtype=np.float32)
        
            intersection = np.sum(y_true_f * y_pred_f)#da por hecho que solo hay dos clases
            union = np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f)
            
            diff_aux_0 = 2.0 * (((y_true_f * union) / (union) ** 2) - (2.0*np.array(bottom[1].data[i, 1, ...]).flatten()*(intersection) / (union) ** 2))              
                    
            diff_aux_1 = -2.0 * (((y_true_f * union) / (union) ** 2) - (2.0*np.array(bottom[1].data[i, 1, ...]).flatten()*(intersection) / (union) ** 2))              
                
            diff_aux_0 = diff_aux_0.reshape(ind.shape)
            diff_aux_1 = diff_aux_1.reshape(ind.shape)    
            
            self.diff1[i,0,:,:] = diff_aux_0
            self.diff1[i,1,:,:] = diff_aux_1
            dice[i] = (2.0 * intersection) / (union)
            
        top[0].data[...]= 1.0 - np.mean(dice)
        print top[0].data[...]
  
    def backward(self, top, propagate_down, bottom):
        bottom[1].diff[...] = self.diff1 #maximizar dice 
        
