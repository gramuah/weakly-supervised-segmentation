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
IMAGE_FILE = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/data/'
IMAGE_FILE_1 = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/data_2/'
IMAGE_FILE_2 = '/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12_2cam_grid_crf/data_3/'
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
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[0].data[...] = 0.0
       
        
    
    def forward(self, bottom, top):
        for c in range(0,bottom[0].data.shape[0]):

            scipy.misc.toimage(bottom[0].data[c, ...], high=255, low=0).save(IMAGE_FILE + 'data_{}.png'.format(self.count))
            scipy.misc.toimage(bottom[1].data[c, ...], high=255, low=0).save(IMAGE_FILE_1 + 'data_2_{}.png'.format(self.count))	
            scipy.misc.toimage(bottom[2].data[c, ...], high=255, low=0).save(IMAGE_FILE_2 + 'data_3_{}.png'.format(self.count))	
            self.count += 1

        top[0].data[...] = bottom[0].data[...]


    def backward(self, top, propagate_down, bottom):
        #pass
        # print "top[0].diff[...] ", top[0].diff[...] = 0.0 esta llegando 0
        bottom[0].diff[...] = 0.0 #top[0].diff[...]
