#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
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
import cv2
from numpy import unravel_index

PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_cam/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'binary_mask/'

class thresholdingLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        if len(bottom) != 3:
            raise Exception("Need 1 input to compute the image label.")
        
    def reshape(self, bottom, top):
        N = 1 #classes + back
        ##self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0
       
        
    
    def forward(self, bottom, top):
        max_value =  np.zeros(shape=(1,bottom[0].data.shape[0]), dtype=np.float32)
        max_value -= 1000000000000000000         
        
        for i in range(0,bottom[0].data.shape[0]):
            for j in range(0,20):
                #aux = cv2.resize(bottom[1].data[i, j, ...], (bottom[0].data.shape[2],bottom[0].data.shape[3]))
                #scipy.misc.toimage(aux, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'map_clc_{}_{}.png'.format(j, self.count))
                heat_map_flatten = bottom[0].data[i, j, ...].flatten() #aux.flatten() #
                max_value_aux = np.max(heat_map_flatten)
                
                
                if (max_value_aux > max_value[0,i]):
                    max_value[0,i] = max_value_aux

        
        
        for i in range(0,bottom[0].data.shape[0]):
            if max_value[0, i] > 0.0:
                thres = max_value[0, i] - max_value[0, i]*0.2
            else:
                thres = max_value[0, i] + max_value[0, i]*0.2
            
            #print self.count, max_value[0, i], thres
            top[0].data[i, 0, ...] = 0.0
            top_aux = np.zeros(shape=(bottom[0].data.shape[2],bottom[0].data.shape[3]), dtype=np.float32)
            for j in range(0,20):
                #print thres
                idx = []
                
                auxi = np.zeros(shape=(bottom[0].data.shape[2]*bottom[0].data.shape[3]), dtype=np.float32)
                auxii = np.zeros(shape=(bottom[0].data.shape[2]*bottom[0].data.shape[3]), dtype=np.float32)
                #aux = cv2.resize(bottom[1].data[i, j, ...], (bottom[0].data.shape[3],bottom[0].data.shape[2]))
                auxi = bottom[0].data[i, j, ...].flatten() #aux.flatten() #
                idx = np.argwhere(auxi > thres)
                
                auxii[idx] = 1.0
                
                top_aux += np.reshape(auxii, (bottom[0].data.shape[2],bottom[0].data.shape[3]))
            
            idx =[]    
            auxb = np.zeros(shape=(bottom[0].data.shape[2]*bottom[0].data.shape[3]), dtype=np.float32)
            auxbb = np.zeros(shape=(bottom[0].data.shape[2]*bottom[0].data.shape[3]), dtype=np.float32)
            auxb = top_aux.flatten()
            
            idx = np.argwhere(auxb > 0.0)
            auxbb[idx] = 1.0
            top[0].data[i, 0, ...] = np.reshape(auxbb, (bottom[0].data.shape[2],bottom[0].data.shape[3]))
            
            # 
            #au = np.argwhere(top[0].data[i, 0, ...] == 0)
            
            #if len(au) >0:
                #scipy.misc.toimage(top[0].data[i, 0, ...], high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + '{}.png'.format(self.count))
                #sio.savemat(IMAGE_FILE_BMMASK_FCN + 'db_{}'.format(self.count),{'bm':top[0].data[i, 0, ...]})
            #else:
                #top[0].data[i, 0, 0, 0] = 0.0
                #scipy.misc.toimage(top[0].data[i, 0, ...], high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + '{}.png'.format(self.count))
                #sio.savemat(IMAGE_FILE_BMMASK_FCN + 'db_{}'.format(self.count),{'bm':top[0].data[i, 0, ...]})
                

            #scipy.misc.toimage(bottom[1].data[i, ...]*255, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'gt_{}.png'.format(self.count))
            #sio.savemat(IMAGE_FILE_BMMASK_FCN + 'prob_{}'.format(self.count),{'prob':bottom[2].data[i, ...]}) 
            #sio.savemat(IMAGE_FILE_BMMASK_FCN + 'max_value_{}'.format(self.count),{'value':max_value[0, i]}) 
            #scipy.misc.toimage(bottom[2].data[i, 0, ...]*255, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'prob_{}.png'.format(self.count))
            self.count += 1
            #print "hola "
            #raw_input()

            
         
    def backward(self, top, propagate_down, bottom):
        #pass
        bottom[0].diff[...] = 0.0
        bottom[1].diff[...] = 0.0
        bottom[2].diff[...] = 0.0
