#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')

import caffe

import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

import random

LABEL_PATH = '/home/carolina/projects/MESMERISE/DeepLab/benchmark_RELEASE/dataset/classlabels_ContextVOC12/'
BATCH_SIZE = 20
N_classes = 20
IMAGE_DIM = 41

class PASCALContextLabelLayer(caffe.Layer):


    def setup(self, bottom, top):
        # config
        self.label_clc = np.zeros((BATCH_SIZE, N_classes))
        
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define 1 top: labels per images.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        
        self.indices = open('/home/carolina/projects/MESMERISE/DeepLab/benchmark_RELEASE/dataset/train_id.txt', 'r').read().splitlines()
        self.idx = 0

    def reshape(self, bottom, top):     
        top[0].reshape(BATCH_SIZE, N_classes)
        top[1].reshape(BATCH_SIZE, IMAGE_DIM, IMAGE_DIM)

    def forward(self, bottom, top):
        h = 0
        self.label_bm = []
        for i in range(0, BATCH_SIZE):
            self.label_clc[i, ...] = 0.0
            
            # load image + label image pair
            if (self.idx + i) > (len(self.indices)-1):
                l_aux = scipy.io.loadmat(LABEL_PATH + self.indices[h] + '.mat')
                
                label_gt = self.load_label(self.indices[h])
                label_aux_bm = np.zeros(shape=(label_gt.shape[0],label_gt.shape[1]), dtype=np.float32)
                auxi = np.zeros(shape=(label_gt.shape[0]*label_gt.shape[1]), dtype=np.float32)
                auxii = np.zeros(shape=(label_gt.shape[0]*label_gt.shape[1]), dtype=np.float32)
                auxi = label_gt.flatten()
                idx = np.argwhere(auxi > 0.0)
                auxii[idx] = 1.0
                label_aux_bm = np.reshape(auxii, (label_gt.shape[0],label_gt.shape[1]))
                h += 1
            else:
                
                l_aux = scipy.io.loadmat(LABEL_PATH + self.indices[self.idx + i] + '.mat')
                label_gt = self.load_label(self.indices[self.idx + i])
                label_aux_bm = np.zeros(shape=(label_gt.shape[0],label_gt.shape[1]), dtype=np.float32)
                auxi = np.zeros(shape=(label_gt.shape[0]*label_gt.shape[1]), dtype=np.float32)
                auxii = np.zeros(shape=(label_gt.shape[0]*label_gt.shape[1]), dtype=np.float32)
                auxi = label_gt.flatten()
                idx = np.argwhere(auxi > 0.0)
                auxii[idx] = 1.0
                label_aux_bm = np.reshape(auxii, (label_gt.shape[0],label_gt.shape[1]))
                        
            lab = np.array(l_aux['label_clc']) 
            
                 
            self.label_clc[i, ... ] = lab
            self.label_bm.append(label_aux_bm) 

                  
            ##self.label = self.crop_label(self.label_res)
            # reshape tops to fit (leading 1 is for batch dimension)
        
        # assign output
        top[0].data[...] = self.label_clc
        top[1].data[...] = self.label_bm
        
        
        # pick next input
        #if self.random:
        #    self.idx = random.randint(0, len(self.indices)-1)
        #else:
        self.idx += BATCH_SIZE
        if self.idx > (len(self.indices)-1):
            self.idx = h

    def backward(self, top, propagate_down, bottom):
        pass
        
        
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 400 labels are translated to the 59 class task labels.
        """
        #label_400 = scipy.io.loadmat('{}/trainval/{}.mat'.format(self.context_dir, idx))['LabelMap']
        #label = np.zeros_like(label_400, dtype=np.uint8)
        #for idx, l in enumerate(self.labels_59):
        #    idx_400 = self.labels_400.index(l) + 1
        #    label[label_400 == idx_400] = idx + 1
        #label = label[np.newaxis, ...]
        label = Image.open('/home/carolina/projects/MESMERISE/DeepLab/benchmark_RELEASE/dataset/SegmentationClassAug/{}.png'.format(idx)) 
        label = np.array(label, dtype=np.float32)
        label_res = scipy.misc.imresize(label, (IMAGE_DIM, IMAGE_DIM), interp='bilinear')

        return label_res  
