#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python/')

import caffe

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
#import cv2

PATH_RES = '/home/carolina/projects/MESMERISE/caffe-master/examples/Segnet-Tutorial/PASCAL/segmentation_results/'
HEATMAP_FILE = PATH_RES + 'cam_heatmap/'

caffe.set_mode_gpu()
caffe.set_device(0)


def grap_solver():
    #solver = caffe.get_solver('solver_CAM_fcn.prototxt')
    solver = caffe.get_solver('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/camvidsiamesas/config/vgg128_noup/solver_train_aug.prototxt')
    solver.net.copy_from('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/camvidsiamesas/model/vgg128_noup/init.caffemodel')
    #solver.net.copy_from('/home/carolina/projects/MESMERISE/caffe-master/examples/Segnet-Tutorial/PASCAL/pretrained/bvlc_caffenet_fcn_cam.caffemodel')
    return solver

def train_model(solver, max_iters = 5000):


    
    
    niter=max_iters
    train_loss = np.zeros(niter)
    
    iter = 0
    while iter < niter :
        solver.step(1)
        #print solver.net.params['u'][0].data[0]
        #print "pesos ", solver.net.params['heatmap_res1'][0].data[3,0, ...]
        #print "pesos ", solver.net.params['heatmap_res1'][0].data[3,1, ...]
        #print "pesos ", solver.net.params['heatmap_res1'][0].data[3,3, ...]
        #print "score_pool4 ", solver.net.params['CAM_fc_pascal'][0].data[0,0]
        
        #print "hola ", solver.net.blobs['CAM_conv_new_map_res'].data[0,0,0,0]
        
        #print "pesos ", solver.net.params['conv1_1'][0].data
        #print "pesos ", solver.net.params['upscore2'][0].data[2,...]
        #print "pesos ", solver.net.params['upscore2'][0].data[20, ...]
        #print "pesos ", solver.net.params['upscore16'][0].data[5,20, ...]
        
        print "pesos_sc ", solver.net.params['conv5_2'][0].data[0, ...]
	raw_input()
        print "pesos_sc ", solver.net.params['fc6'][0].data[0, ...]
	raw_input()
        #print "debug woreshape ", solver.net.blobs['out'].data[19,1]
        print "siam ", solver.net.params['fc6_siam'][0].data[0, ...]
	raw_input()
	print "siam ", solver.net.params['fc8_cam_siam'][0].data[0, ...]
        #print "score ", solver.net.blobs['score'].data[...].shape
        #print "debug reshape ", solver.net.blobs['score'].data[0, ...]
        
        #print "debug score ", solver.net.blobs['score'].data[1, ...]
        
        raw_input()

    #plt.plot(np.arange(niter), train_loss)
    #plt.show()
    

if __name__ == '__main__':

    # Init net
    solver = grap_solver()
    
 
    train_model(solver, max_iters = 5000)
    
  
