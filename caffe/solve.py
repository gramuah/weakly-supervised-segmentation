#!/usr/bin/env python
import sys
sys.path.append('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/python')
import caffe
#import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
    
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

#weights = '/home/carolina/projects/MESMERISE/caffe-master/examples/Segnet-Tutorial/PASCAL/pretrained/VGG_ILSVRC_16_layers.caffemodel'

print "hola0"

net = caffe.Net('/home/carolina/projects/MESMERISE/DeepLab/deeplab-public-ver2/voc12/config/vgg128_noup/test_val.prototxt', 
                '/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/voc12_2cam_grid_crf_fc7_2_switch_input_mirror_fc7/model/vgg128_noup/best_model/init.caffemodel', 
                caffe.TEST)
print "hola1"
net2 = caffe.Net('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/vgg/deploy.prototxt', 
                '/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/vgg/VGG_ILSVRC_16_layers.caffemodel', 
                caffe.TEST)

print "hola2"
solver = caffe.SGDSolver('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/voc12_2cam_grid_crf_fc7_2_switch_input_mirror_fc7/config/vgg128_noup/solver_train_aug.prototxt') #'/home/carolina/projects/MESMERISE/caffe-master/examples/Segnet-Tutorial/PASCAL/CAMmodels/voc2012/CAM_vgg16_iter_100000.caffemodel',

print "holaa"

solver.net.params['conv1_1_d'][0].data[...] =  net.params['conv1_1'][0].data[...]
solver.net.params['conv1_1_d'][1].data[...] =  net.params['conv1_1'][1].data[...]

solver.net.params['conv1_2_d'][0].data[...] =  net.params['conv1_2'][0].data[...]
solver.net.params['conv1_2_d'][1].data[...] =  net.params['conv1_2'][1].data[...]

solver.net.params['conv2_1_d'][0].data[...] =  net.params['conv2_1'][0].data[...]
solver.net.params['conv2_1_d'][1].data[...] =  net.params['conv2_1'][1].data[...]

solver.net.params['conv2_2_d'][0].data[...] =  net.params['conv2_2'][0].data[...]
solver.net.params['conv2_2_d'][1].data[...] =  net.params['conv2_2'][1].data[...]    

solver.net.params['conv3_1_d'][0].data[...] =  net.params['conv3_1'][0].data[...]
solver.net.params['conv3_1_d'][1].data[...] =  net.params['conv3_1'][1].data[...]

solver.net.params['conv3_2_d'][0].data[...] =  net.params['conv3_2'][0].data[...]
solver.net.params['conv3_2_d'][1].data[...] =  net.params['conv3_2'][1].data[...]

solver.net.params['conv3_3_d'][0].data[...] =  net.params['conv3_3'][0].data[...]
solver.net.params['conv3_3_d'][1].data[...] =  net.params['conv3_3'][1].data[...]

solver.net.params['conv4_1_d'][0].data[...] =  net.params['conv4_1'][0].data[...]
solver.net.params['conv4_1_d'][1].data[...] =  net.params['conv4_1'][1].data[...]

solver.net.params['conv4_2_d'][0].data[...] =  net.params['conv4_2'][0].data[...]
solver.net.params['conv4_2_d'][1].data[...] =  net.params['conv4_2'][1].data[...]

solver.net.params['conv4_3_d'][0].data[...] =  net.params['conv4_3'][0].data[...]
solver.net.params['conv4_3_d'][1].data[...] =  net.params['conv4_3'][1].data[...]

solver.net.params['conv5_1_d'][0].data[...] =  net.params['conv5_1'][0].data[...]
solver.net.params['conv5_1_d'][1].data[...] =  net.params['conv5_1'][1].data[...]

solver.net.params['conv5_2_d'][0].data[...] =  net.params['conv5_2'][0].data[...]
solver.net.params['conv5_2_d'][1].data[...] =  net.params['conv5_2'][1].data[...]

solver.net.params['conv5_3_d'][0].data[...] =  net.params['conv5_3'][0].data[...]
solver.net.params['conv5_3_d'][1].data[...] =  net.params['conv5_3'][1].data[...]

solver.net.params['fc6'][0].data[...] =  net.params['fc6'][0].data[...]
solver.net.params['fc6'][1].data[...] =  net.params['fc6'][1].data[...]

solver.net.params['fc7'][0].data[...] =  net.params['fc7'][0].data[...]
solver.net.params['fc7'][1].data[...] =  net.params['fc7'][1].data[...]

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1024)

solver.net.params['fc8_pas'][0].data[0, ...] =  s
solver.net.params['fc8_pas'][1].data[0, ...] =  0.0
solver.net.params['fc8_pas'][0].data[1, ..., 0, 0] =  net2.params['fc8'][0].data[229, ...]
solver.net.params['fc8_pas'][1].data[1, ...] =  net2.params['fc8'][1].data[229, ...]
print solver.net.params['fc8_pas'][1].data[1, ...]
raw_input()
solver.net.params['fc8_pas'][0].data[2, ..., 0, 0] =  net2.params['fc8'][0].data[254, 0:1024]
solver.net.params['fc8_pas'][1].data[2, ...] =  net2.params['fc8'][1].data[254, ...]
solver.net.params['fc8_pas'][0].data[3, ..., 0, 0] =  net2.params['fc8'][0].data[382, 0:1024]
solver.net.params['fc8_pas'][1].data[3, ...] =  net2.params['fc8'][1].data[382, ...]
solver.net.params['fc8_pas'][0].data[4, ..., 0, 0] =  net2.params['fc8'][0].data[235, 0:1024]
solver.net.params['fc8_pas'][1].data[4, ...] =  net2.params['fc8'][1].data[235, ...]
solver.net.params['fc8_pas'][0].data[5, ..., 0, 0] =  net2.params['fc8'][0].data[830, 0:1024]
solver.net.params['fc8_pas'][1].data[5, ...] =  net2.params['fc8'][1].data[830, ...]
solver.net.params['fc8_pas'][0].data[6, ..., 0, 0] =  net2.params['fc8'][0].data[919, 0:1024]
solver.net.params['fc8_pas'][1].data[6, ...] =  net2.params['fc8'][1].data[919, ...]
solver.net.params['fc8_pas'][0].data[7, ..., 0, 0] =  net2.params['fc8'][0].data[273, 0:1024]
solver.net.params['fc8_pas'][1].data[7, ...] =  net2.params['fc8'][1].data[273, ...]
solver.net.params['fc8_pas'][0].data[8, ..., 0, 0] =  net2.params['fc8'][0].data[9, 0:1024]
solver.net.params['fc8_pas'][1].data[8, ...] =  net2.params['fc8'][1].data[9, ...]
solver.net.params['fc8_pas'][0].data[9, ..., 0, 0] =  net2.params['fc8'][0].data[308, 0:1024]
solver.net.params['fc8_pas'][1].data[9, ...] =  net2.params['fc8'][1].data[308, ...]
solver.net.params['fc8_pas'][0].data[10, ..., 0, 0] =  net2.params['fc8'][0].data[60, 0:1024]
solver.net.params['fc8_pas'][1].data[10, ...] =  net2.params['fc8'][1].data[60, ...]
solver.net.params['fc8_pas'][0].data[11, ..., 0, 0] =  net2.params['fc8'][0].data[314, 0:1024]
solver.net.params['fc8_pas'][1].data[11, ...] =  net2.params['fc8'][1].data[314, ...]
solver.net.params['fc8_pas'][0].data[12, ..., 0, 0] =  net2.params['fc8'][0].data[59, 0:1024]
solver.net.params['fc8_pas'][1].data[12, ...] =  net2.params['fc8'][1].data[59, ...]
solver.net.params['fc8_pas'][0].data[13, ..., 0, 0] =  net2.params['fc8'][0].data[292, 0:1024]
solver.net.params['fc8_pas'][1].data[13, ...] =  net2.params['fc8'][1].data[292, ...]
solver.net.params['fc8_pas'][0].data[14, ..., 0, 0] =  net2.params['fc8'][0].data[259, 0:1024]
solver.net.params['fc8_pas'][1].data[14, ...] =  net2.params['fc8'][1].data[259, ...]
solver.net.params['fc8_pas'][0].data[15, ..., 0, 0] =  net2.params['fc8'][0].data[953, 0:1024]
solver.net.params['fc8_pas'][1].data[15, ...] =  net2.params['fc8'][1].data[953, ...]
solver.net.params['fc8_pas'][0].data[16, ..., 0, 0] =  net2.params['fc8'][0].data[837, 0:1024]
solver.net.params['fc8_pas'][1].data[16, ...] =  net2.params['fc8'][1].data[837, ...]
solver.net.params['fc8_pas'][0].data[17, ..., 0, 0] =  net2.params['fc8'][0].data[80, 0:1024]
solver.net.params['fc8_pas'][1].data[17, ...] =  net2.params['fc8'][1].data[80, ...]
solver.net.params['fc8_pas'][0].data[18, ..., 0, 0] =  net2.params['fc8'][0].data[310, 0:1024]
solver.net.params['fc8_pas'][1].data[18, ...] =  net2.params['fc8'][1].data[310, ...]
solver.net.params['fc8_pas'][0].data[19, ..., 0, 0] =  net2.params['fc8'][0].data[886, 0:1024]
solver.net.params['fc8_pas'][1].data[19, ...] =  net2.params['fc8'][1].data[886, ...]
solver.net.params['fc8_pas'][0].data[20, ..., 0, 0] =  net2.params['fc8'][0].data[868, 0:1024]
solver.net.params['fc8_pas'][1].data[20, ...] =  net2.params['fc8'][1].data[868, ...]

solver.net.params['conv1_1'][0].data[...] =  net.params['conv1_1'][0].data[...]
solver.net.params['conv1_1'][1].data[...] =  net.params['conv1_1'][1].data[...]

solver.net.params['conv1_2'][0].data[...] =  net.params['conv1_2'][0].data[...]
solver.net.params['conv1_2'][1].data[...] =  net.params['conv1_2'][1].data[...]

solver.net.params['conv2_1'][0].data[...] =  net.params['conv2_1'][0].data[...]
solver.net.params['conv2_1'][1].data[...] =  net.params['conv2_1'][1].data[...]

solver.net.params['conv2_2'][0].data[...] =  net.params['conv2_2'][0].data[...]
solver.net.params['conv2_2'][1].data[...] =  net.params['conv2_2'][1].data[...]    

solver.net.params['conv3_1'][0].data[...] =  net.params['conv3_1'][0].data[...]
solver.net.params['conv3_1'][1].data[...] =  net.params['conv3_1'][1].data[...]

solver.net.params['conv3_2'][0].data[...] =  net.params['conv3_2'][0].data[...]
solver.net.params['conv3_2'][1].data[...] =  net.params['conv3_2'][1].data[...]

solver.net.params['conv3_3'][0].data[...] =  net.params['conv3_3'][0].data[...]
solver.net.params['conv3_3'][1].data[...] =  net.params['conv3_3'][1].data[...]

solver.net.params['conv4_1'][0].data[...] =  net.params['conv4_1'][0].data[...]
solver.net.params['conv4_1'][1].data[...] =  net.params['conv4_1'][1].data[...]

solver.net.params['conv4_2'][0].data[...] =  net.params['conv4_2'][0].data[...]
solver.net.params['conv4_2'][1].data[...] =  net.params['conv4_2'][1].data[...]

solver.net.params['conv4_3'][0].data[...] =  net.params['conv4_3'][0].data[...]
solver.net.params['conv4_3'][1].data[...] =  net.params['conv4_3'][1].data[...]

solver.net.params['conv5_1'][0].data[...] =  net.params['conv5_1'][0].data[...]
solver.net.params['conv5_1'][1].data[...] =  net.params['conv5_1'][1].data[...]

solver.net.params['conv5_2'][0].data[...] =  net.params['conv5_2'][0].data[...]
solver.net.params['conv5_2'][1].data[...] =  net.params['conv5_2'][1].data[...]

solver.net.params['conv5_3'][0].data[...] =  net.params['conv5_3'][0].data[...]
solver.net.params['conv5_3'][1].data[...] =  net.params['conv5_3'][1].data[...]

solver.net.params['fc6_p'][0].data[...] =  net.params['fc6'][0].data[...]
solver.net.params['fc6_p'][1].data[...] =  net.params['fc6'][1].data[...]

solver.net.params['fc7_p'][0].data[...] =  net.params['fc7'][0].data[...]
solver.net.params['fc7_p'][1].data[...] =  net.params['fc7'][1].data[...]


solver.net.params['conv1_1_c'][0].data[...] =  net.params['conv1_1'][0].data[...]
solver.net.params['conv1_1_c'][1].data[...] =  net.params['conv1_1'][1].data[...]

solver.net.params['conv1_2_c'][0].data[...] =  net.params['conv1_2'][0].data[...]
solver.net.params['conv1_2_c'][1].data[...] =  net.params['conv1_2'][1].data[...]

solver.net.params['conv2_1_c'][0].data[...] =  net.params['conv2_1'][0].data[...]
solver.net.params['conv2_1_c'][1].data[...] =  net.params['conv2_1'][1].data[...]

solver.net.params['conv2_2_c'][0].data[...] =  net.params['conv2_2'][0].data[...]
solver.net.params['conv2_2_c'][1].data[...] =  net.params['conv2_2'][1].data[...]    

solver.net.params['conv3_1_c'][0].data[...] =  net.params['conv3_1'][0].data[...]
solver.net.params['conv3_1_c'][1].data[...] =  net.params['conv3_1'][1].data[...]

solver.net.params['conv3_2_c'][0].data[...] =  net.params['conv3_2'][0].data[...]
solver.net.params['conv3_2'][1].data[...] =  net.params['conv3_2'][1].data[...]

solver.net.params['conv3_3_c'][0].data[...] =  net.params['conv3_3'][0].data[...]
solver.net.params['conv3_3_c'][1].data[...] =  net.params['conv3_3'][1].data[...]

solver.net.params['conv4_1_c'][0].data[...] =  net.params['conv4_1'][0].data[...]
solver.net.params['conv4_1_c'][1].data[...] =  net.params['conv4_1'][1].data[...]

solver.net.params['conv4_2_c'][0].data[...] =  net.params['conv4_2'][0].data[...]
solver.net.params['conv4_2_c'][1].data[...] =  net.params['conv4_2'][1].data[...]

solver.net.params['conv4_3_c'][0].data[...] =  net.params['conv4_3'][0].data[...]
solver.net.params['conv4_3_c'][1].data[...] =  net.params['conv4_3'][1].data[...]

solver.net.params['conv5_1_c'][0].data[...] =  net.params['conv5_1'][0].data[...]
solver.net.params['conv5_1_c'][1].data[...] =  net.params['conv5_1'][1].data[...]

solver.net.params['conv5_2_c'][0].data[...] =  net.params['conv5_2'][0].data[...]
solver.net.params['conv5_2_c'][1].data[...] =  net.params['conv5_2'][1].data[...]

solver.net.params['conv5_3_c'][0].data[...] =  net.params['conv5_3'][0].data[...]
solver.net.params['conv5_3_c'][1].data[...] =  net.params['conv5_3'][1].data[...]


solver.net.params['fc6_p_c'][0].data[...] =  net.params['fc6'][0].data[...]
solver.net.params['fc6_p_c'][1].data[...] =  net.params['fc6'][1].data[...]

solver.net.params['fc7_p_c'][0].data[...] =  net.params['fc7'][0].data[...]
solver.net.params['fc7_p_c'][1].data[...] =  net.params['fc7'][1].data[...]

print "hola ", solver.net.params['fc8_pas'][0].data[4, ..., 0, 0]
raw_input()
print solver.net.params['fc8_pas'][1].data[4, ...]
raw_input()

solver.net.save('/home/carolina/projects/MESMERISE/DeepLab/deeplab-ferrari/deeplab/voc12_2cam_grid_crf_fc7_2_switch_input_mirror_fc7/model/vgg128_noup/init_new.caffemodel')




