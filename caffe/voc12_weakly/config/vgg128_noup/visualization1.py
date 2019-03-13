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
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/gt41/'
PATH_RES = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/'
IMAGE_FILE_BMMASK_FCN = PATH_RES + 'back_crf/'
IMAGE_FILE_BMMASK_FCN_1 = PATH_RES + 'conv5_new/'
IMAGE_FILE_BMMASK_FCN_2 = PATH_RES + 'map/'

class VisLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self.count = 0
        self.epoch = 0
        if len(bottom) != 2:
            raise Exception("Need 2 input to compute the image label.")
        
    def reshape(self, bottom, top):	
        top[0].reshape(bottom[0].data.shape[0], 21, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0
       
        
    
    def forward(self, bottom, top):
        for c in range(0,bottom[0].data.shape[0]):
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
    
            if (self.count == 15) or (self.count == 10597) or (self.count == 21179) or (self.count == 31761) or (self.count == 42343) or (self.count == 52925) or (self.count == 63507) or (self.count == 74089) or (self.count == 84671) or (self.count == 95253) or (self.count == 105835):
	            #aux_save = scipy.misc.imresize(rgb, (200,200)) 
	            scipy.misc.toimage(rgb, high=255, low=0).save(IMAGE_FILE + 'img_{}.png'.format(self.epoch))
	            self.epoch += 1
            
            self.count += 1


    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0 #top[0].diff[...]
