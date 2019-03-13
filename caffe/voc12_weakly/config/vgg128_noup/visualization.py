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

from numpy import unravel_index

IMAGE_FILE_MAT_PROB = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/prob_mat/'
IMAGE_FILE_MAT = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/voc12_2cam_grid_crf_fc7_2_switch_input_mirror/after/'
IMAGE_FILE_MAT2 = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/voc12_2cam_grid_crf_fc7_2_switch_input_mirror/gt/'
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/voc12_2cam_grid_crf_fc7_2_switch_input_mirror/before/'
PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'back_crf/'
IMAGE_FILE_BMMASK_FCN_1 = PATH_RES + 'conv5_new/'
IMAGE_FILE_BMMASK_FCN_2 = PATH_RES + 'map/'

class VisLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        self.epoch = 0
        if len(bottom) != 3:
            raise Exception("Need 2 input to compute the image label.")
        
    def reshape(self, bottom, top):	
        top[0].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0
       
        
    
    def forward(self, bottom, top):
        for c in range(0,bottom[0].data.shape[0]):
	    #print "data0 ", bottom[0].data[c, ...]
	    #print "data1 ", bottom[1].data[c, ...]
            #raw_input()
            ind_a = np.argmax(bottom[0].data[c, ...], axis=0)
	    #ind_a = np.copy(bottom[0].data[c, ...])
	    ind = np.squeeze(ind_a)
	    
            r = ind.copy()
            g = ind.copy()
            b = ind.copy()

	    ind_a_gt = np.argmax(bottom[1].data[c, ...], axis=0)
	    #ind_a_gt = np.copy(bottom[1].data[c, ...])
	    ind_gt = np.squeeze(ind_a_gt)
	    
            r_gt = ind_gt.copy()
            g_gt = ind_gt.copy()
            b_gt = ind_gt.copy()

	    #ind_a_gtt = np.argmax(bottom[2].data[c, ...], axis=0)
	    ind_a_gtt = np.copy(bottom[2].data[c, ...])
	    ind_gtt = np.squeeze(ind_a_gtt)
	    
            r_gtt = ind_gtt.copy()
            g_gtt = ind_gtt.copy()
            b_gtt = ind_gtt.copy()
            
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

		        r_gt[ind_gt==l] = label_colours[l,0]
		        g_gt[ind_gt==l] = label_colours[l,1]
		        b_gt[ind_gt==l] = label_colours[l,2]

		        r_gtt[ind_gtt==l] = label_colours[l,0]
		        g_gtt[ind_gtt==l] = label_colours[l,1]
		        b_gtt[ind_gtt==l] = label_colours[l,2]
		        
            rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	 
            rgb[:,:,0] = r/255.0
            rgb[:,:,1] = g/255.0
            rgb[:,:,2] = b/255.0

            rgb_gt = np.zeros((ind_gt.shape[0], ind_gt.shape[1], 3))
	 
            rgb_gt[:,:,0] = r_gt/255.0
            rgb_gt[:,:,1] = g_gt/255.0
            rgb_gt[:,:,2] = b_gt/255.0    
            
	    rgb_gtt = np.zeros((ind_gtt.shape[0], ind_gtt.shape[1], 3))
	 
            rgb_gtt[:,:,0] = r_gtt/255.0
            rgb_gtt[:,:,1] = g_gtt/255.0
            rgb_gtt[:,:,2] = b_gtt/255.0 
            #if (self.count == 15) or (self.count == 10597) or (self.count == 21179) or (self.count == 31761) or (self.count == 42343) or (self.count == 52925) or (self.count == 63507) or (self.count == 74089) or (self.count == 84671) or (self.count == 95253) or (self.count == 105835):
	            #aux_save = scipy.misc.imresize(rgb, (200,200)) 
            scipy.misc.toimage(rgb, high=255, low=0).save(IMAGE_FILE + 'before_{}.png'.format(self.count))
	    scipy.misc.toimage(rgb_gt, high=255, low=0).save(IMAGE_FILE_MAT + 'after_{}.png'.format(self.count))
	    scipy.misc.toimage(rgb_gtt, high=255, low=0).save(IMAGE_FILE_MAT2 + 'gt_{}.png'.format(self.count))
            	#self.epoch += 1
            
            self.count += 1


    def backward(self, top, propagate_down, bottom):
        pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        #bottom[0].diff[...] = 0.0 #top[0].diff[...]
