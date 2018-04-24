import numpy as np
import pandas as pd
import kernel_descriptors as kd
import paint_auth as pa
import cv2

"""
Tests for kernel mapping approximation
"""
dimension = 500
z = np.random.rand(dimension)
d = np.random.rand(2000,dimension)

norm = np.linalg.norm(d, axis=1)
for i in range(len(d)):
    d[i] = d[i] / norm[i]
    
truth = np.array([kd.gaussian_kernel(d[i],z) for i in range(len(d))])

##D > d test
#ratio = 10
#error = []
#for i in range(ratio):
    #estimate = []
    #D = dimension * (i + 1)
    #for i in range(len(d)):
        #estimate.append(np.matmul(kd.rbf_map(d[i], D), kd.rbf_map(d[i], D)))
    #estimate = np.array(estimate)
    #error.append(np.square((truth-estimate)).mean())
    
    
    
## D <= d test
dim = [600,700,800]
error = []
for D in dim:
    estimate = []
    for i in range(len(d)):
        estimate.append(np.matmul(kd.rbf_map(d[i], D), kd.rbf_map(d[i], D)))
    estimate = np.array(estimate)
    error.append(np.square((truth-estimate)).mean())  