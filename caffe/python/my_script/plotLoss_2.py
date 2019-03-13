#!/usr/bin/env python

'''
Created on Jan 5, 2016

@author: Daniel Onoro Rubio
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

key = ", loss ="
key2 = " #0: accuracy ="
key1 = ": euclidean_loss ="
key3 = ": smx_loss ="
if __name__ == '__main__':
    log_file = sys.argv[1]
    log_file2 = sys.argv[2]
   
    print "Processing log file: ", log_file
    loss_v = []
    it_v = []
    it_v2 = []
    loss_v2 = []
    loss_euc = []
    loss_smx = []
    with open(log_file,'r') as f:
        a = 0
        jump = 20
        for line in f:
            if key in line:
                s = line.split(' ')
                
                a += jump
                it_v.append( int(a) )
                loss_v.append( float( s[8].strip())  )
            #if key2 in line:
            #    ss = line.split(' ') 
            #    loss_v2.append( float(ss[14].strip()) )
    
    with open(log_file2,'r') as f2:
        a = 0
        jump = 20
        for line in f2:
            if key in line:
                s = line.split(' ')
                
                loss_v2.append( float( s[8].strip()))
            if key1 in line:
                s = line.split(' ')
                
                
                
                loss_euc.append( float( s[14].strip()) )
            if key2 in line:
                s = line.split(' ')
                
                a += jump
                it_v2.append( int(a) )
                loss_smx.append( float( s[14].strip()) )

    
    print len(it_v2), len(loss_v2)
    
    plt.plot(it_v, loss_v, linewidth=2.0)
    plt.plot(it_v2, loss_v2, linewidth=2.0)
    plt.plot(it_v2, loss_euc, linewidth=2.0)
    plt.plot(it_v2, loss_smx, linewidth=2.0)
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.axis([0, 14000, 0, 7])
    
    plt.grid(True)
    plt.show()
