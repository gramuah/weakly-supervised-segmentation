#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.io as sio
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

IMAGE_FILE_MAT_PROB = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/prob_mat/'
IMAGE_FILE_MAT = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/heat_mat/'
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/gt_heat/'
PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'mask_new/'
IMAGE_FILE_BMMASK_FCN_1 = PATH_RES + 'conv5_new/'
IMAGE_FILE_BMMASK_FCN_2 = PATH_RES + 'map/'
IMAGE_FILE_BMMASK_FCN_4 = PATH_RES + 'movida/'
class NormLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        self.epoch = 0
        if len(bottom) != 6:
            raise Exception("Need 5 input to compute the image label.")
        
    def reshape(self, bottom, top):
        N = 20 #classes + back
        ##self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[1].reshape(bottom[0].data.shape[0], N+1, bottom[0].data.shape[2], bottom[0].data.shape[3])
	top[2].reshape(bottom[0].data.shape[0], 1, 1, 2)
        #top[0].data[...] = 1.0
	top[1].data[...] = 0.0
       
        
    
    def forward(self, bottom, top):

        #db
	#print bottom[5].data[...].shape
	#print bottom[5].data[...]
        #raw_input()
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

        
        for i in range(0,bottom[0].data.shape[0]):
	    top[2].data[i, 0, 0, ...] = 41
            top[0].data[i, 0, ...] = 1.0
            aux_c = np.sum(bottom[1].data[i, ...], axis = 0)
            #aux_save_1 = scipy.misc.imresize(aux_c, (100,100))
            #scipy.misc.toimage(aux_save_1, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_2 + 'map_clc_{}.png'.format(self.count))

            aux_cc = scipy.misc.imresize(np.sum(bottom[2].data[i, ...], axis = 0), (14,14))
            #aux_save_2 = scipy.misc.imresize(aux_cc, (100,100))
            #scipy.misc.toimage(aux_save_2, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_1 + 'map_clc_{}.png'.format(self.count))
            aux_c5 = np.sum(bottom[3].data[i, ...], axis = 0)
            
            aux_ccc = aux_c + aux_cc + aux_c5 # 
            aux_ccc = (aux_ccc - np.min(aux_ccc))/float(np.max(aux_ccc)- np.min(aux_ccc))# 
            #scipy.misc.toimage(aux_ccc, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_4 + 'map_clc_{}.png'.format(self.count))
            
	    aux2 = np.zeros(shape=(14*14), dtype=np.float32)
            aux3 = np.zeros(shape=(14,14), dtype=np.float32)
	    bm = np.zeros(shape=(14,14), dtype=np.float32)
            bm_aux = np.zeros(shape=(14*14), dtype=np.float32)
	    auxi = aux_ccc.flatten()
            idx = []
	    idx2 = []
            idx = np.argwhere(auxi < 0.3*np.max(aux_ccc))
            idx2 = np.argwhere(auxi > 0.8*np.max(aux_ccc))
            aux2[idx] = 1.0
	    bm_aux[idx2] = 1.0
	    bm = np.reshape(bm_aux, (14,14))
            aux3 = np.reshape(aux2, (14,14))
            top[0].data[i, 0, ...] -= aux_ccc #aux3
	    #patata = top[0].data[i, 0, ...]
	    #scipy.misc.toimage(patata, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_4 + 'patata_{}.png'.format(self.count))
            
	    if (self.count == 15) or (self.count == 10597) or (self.count == 21179) or (self.count == 31761) or (self.count == 42343) or (self.count == 52925) or (self.count == 63507) or (self.count == 74089) or (self.count == 84671) or (self.count == 95253):
            	aux_save = scipy.misc.imresize(aux3, (200,200))
            	aux_save_2 = scipy.misc.imresize(aux_cc, (200,200))
            	scipy.misc.toimage(aux_save, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'img_{}.png'.format(self.epoch))
            	scipy.misc.toimage(aux_save_2, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_1 + 'img_{}.png'.format(self.epoch))
            #scipy.misc.toimage(aux_ccc, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_1 + 'map_clc_{}.png'.format(self.count))
            #self.count +=1
            #print aux_cc
            #raw_input()
            #self.count +=1
            #print self.count
            max_value = -1000000000000000000000000000
	    min_value = 1000000000000000000000000000
            #aux = np.zeros(shape=(14,14), dtype=np.float32)
	    aux_sum = np.zeros(shape=(14,14), dtype=np.float32)
            #aux2 = np.zeros(shape=(14,14), dtype=np.float32)
            #aux3 = np.zeros(shape=(14*14), dtype=np.float32)
            #aux4 = np.zeros(shape=(14*14), dtype=np.float32)
            db = 0
            for j in range(0,bottom[0].data.shape[1]):
		if bottom[5].data[i,j]:
                	heat_map_flatten = bottom[0].data[i, j, ...].flatten()#bottom[4].data[i,j]*(
                	max_value_aux = np.max(heat_map_flatten)
                	min_value_aux = np.min(heat_map_flatten)
                	if (max_value_aux > max_value):
                    		max_value = max_value_aux
                    		db = j
                	if (min_value_aux < min_value):
                    		min_value = min_value_aux
  
            if (self.count == 15) or (self.count == 10597) or (self.count == 21179) or (self.count == 31761) or (self.count == 42343) or (self.count == 52925) or (self.count == 63507) or (self.count == 74089) or (self.count == 84671) or (self.count == 95253):
	            ind = []	
	            ind = np.argmax(bottom[0].data[i, ...], axis=0)
	            r = ind.copy()
	            g = ind.copy()
	            b = ind.copy()
            
	            label_colours_map = np.array([aeroplane,  bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor])
	            for l in range(0,20):
		        	r[ind==l] = label_colours_map[l,0]
		        	g[ind==l] = label_colours_map[l,1]
		        	b[ind==l] = label_colours_map[l,2]
		        
	            rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	            rgb[:,:,0] = r/255.0
	            rgb[:,:,1] = g/255.0
	            rgb[:,:,2] = b/255.0


	            aux_save_3 = scipy.misc.imresize(rgb, (200,200)) 
	            scipy.misc.toimage(aux_save_3, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN_2 + 'img_{}.png'.format(self.epoch))
	            


            
            
	    #print bottom[4].data[i, ...]
	    #raw_input()
	    #print bottom[5].data[i,j]
            for j in range(0,bottom[0].data.shape[1]):
		if (bottom[5].data[i,j] == 0.0):#np.max(bottom[0].data[i, j, ...])
		      	top[1].data[i, j+1, ...] = 0.0

	        else:
			#thres = 0.2*np.max(bottom[0].data[i, j, ...])
			#auxi = bottom[0].data[i, j, ...].flatten()
			#idx = []
			#idx = np.argwhere(auxi < thres)
			#aux3[idx] = 1.0
			#aux = np.reshape(aux3, (14,14)) 
			#aux2 = 0.5*(aux * aux_ccc) + 0.5*bottom[0].data[i, j, ...]
			#aux_sum += bottom[0].data[i, j, ...]
			aux_sum = bottom[0].data[i, j, ...]		
			top[1].data[i, j+1, ...] = (aux_sum- np.min(aux_sum) )/float(np.max(aux_sum)- np.min(aux_sum) )#   
			#top[1].data[i, j+1, ...] = (bottom[0].data[i, j, ...] - np.min(bottom[0].data[i, j, ...]))/float(np.max(bottom[0].data[i, j, ...]) - np.min(bottom[0].data[i, j, ...]))#
			#top[1].data[i, j+1, ...] = (bottom[0].data[i, j, ...] - min_value)/float(max_value - min_value)#
		
            #m_0 = 1 - (1/float(bottom[0].data.shape[1])*aux_sum)
	    #top[0].data[i, 0, ...] = 0.5*top[0].data[i, 0, ...] + 0.5*m_0
	    #top[0].data[i, 0, ...] = (top[0].data[i, 0, ...] - np.min(top[0].data[i, 0, ...]))/float(np.max(top[0].data[i, 0, ...]) - np.min(top[0].data[i, 0, ...]))     
            top[1].data[i, 0, ...] = top[0].data[i, 0, ...]
	    #print "aux ", aux
            #aux2 = np.argmax(top[0].data[i, ...], axis=0)
            #print aux2
            #raw_input()
            
            #scipy.misc.toimage(top[0].data[i, 0, ...], high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'map_clc_{}.png'.format(self.count))
            #self.count +=1
            #scipy.misc.toimage(top[0].data[i, 0, ...], high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'map_clc_{}.png'.format(self.count))
            #self.count +=1
            #scipy.misc.toimage(aux, high=255, low=0).save(IMAGE_FILE_BMMASK_FCN + 'map_clc_{}.png'.format(self.count))
            #self.count +=1

   
            if (self.count == 15) or (self.count == 10597) or (self.count == 21179) or (self.count == 31761) or (self.count == 42343) or (self.count == 52925) or (self.count == 63507) or (self.count == 74089) or (self.count == 84671) or (self.count == 95253):
	            ##Plot the mask
	            sio.savemat(IMAGE_FILE_MAT + 'img_{}.mat'.format(self.epoch),{'pred':bottom[0].data[i,...]})
	            sio.savemat(IMAGE_FILE_MAT_PROB + 'prob_{}.mat'.format(self.epoch),{'prob':bottom[4].data[i,...]})
	            ind = []	
	            ind = np.argmax(top[1].data[i, ...], axis=0)
	            r = ind.copy()
	            g = ind.copy()
	            b = ind.copy()

	            label_colours = np.array([background, aeroplane,  bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor])
	            for l in range(0,21):
		        	r[ind==l] = label_colours[l,0]
		        	g[ind==l] = label_colours[l,1]
		        	b[ind==l] = label_colours[l,2]
		        
	            rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	            rgb[:,:,0] = r/255.0
	            rgb[:,:,1] = g/255.0
	            rgb[:,:,2] = b/255.0
	            aux_save = scipy.misc.imresize(rgb, (200,200)) 
	            scipy.misc.toimage(aux_save, high=255, low=0).save(IMAGE_FILE + 'img_{}.png'.format(self.epoch))
	            self.epoch += 1
            
	    
            self.count += 1
            
    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0 #top[0].diff[...]
