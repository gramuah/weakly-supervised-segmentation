#!/usr/bin/env python

'''
Created on Jan 5, 2016

@author: Daniel Onoro Rubio
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

key = ", loss ="
key2 = " #1: accuracy ="
key3 = " #3: euclidean_loss ="
key4 = " #4: euclidean_loss_c ="

if __name__ == '__main__':
    log_file = sys.argv[1]
    
    print "Processing log file: ", log_file
    loss_v = []
    it_v = []
    loss_v2 = []
    loss_v3 = []
    loss_v4 = []
    with open(log_file,'r') as f:
        #a = 100000
	a = 0        
	jump = 20
        for line in f:
            if key in line:
                s = line.split(' ')
		
                
                it_v.append( int(a) )
		a += jump         
		if (len(s) == 9):
                    loss_v.append( float( s[8].strip()) )
                else:
                    loss_v.append( float( s[len(s)-1].strip()) )
            if key2 in line:
                ss = line.split(' ') 
                loss_v2.append( float(ss[14].strip()) )
            if key3 in line:
                sss = line.split(' ') 
                loss_v3.append( float(sss[14].strip()) )
            if key4 in line:
                ssss = line.split(' ') 
                loss_v4.append( float(ssss[14].strip()) )

    print len(loss_v3), len(it_v)
             
    plt.plot(it_v, loss_v, linewidth=2.0)
    #plt.plot(it_v, loss_v2, linewidth=2.0)
    #plt.plot(it_v, loss_v3, linewidth=2.0)
    #plt.plot(it_v, loss_v4, linewidth=2.0)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.axis([0, 100000, 0, 7.5])
    plt.grid(True)
    plt.show()
