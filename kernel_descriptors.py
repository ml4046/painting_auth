"""
Contains kernel descriptors and tamura kernel descriptors
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_gradient_magnitude

def gaussian_kernel(x,y, gamma=0.5):
    """
    Computes spatial difference between x,y with gaussian kernel
    x: array of values
    y: array of values
    gamma: float hyperparam in kernel
    
    return:
    val: float numerical value of the kernel value
    """
    return np.exp(-gamma * np.square(np.linalg.norm((abs(x-y)))))
def jlt_map(x, k):
    d = len(x)
    A = orf_matrix((k,d),sigma=1)
    return (1.0 / np.sqrt(k)) * np.matmul(A, x)

def kdes_grad(img, D=16, k=16, sigma=1):
    """
    Computes the gradient kernel using orf matrix
    
    Feature map approximated with random ortho gaussian matrix
    """
    m, n = img.shape
    z = img.flatten()
    mag, theta = gradient(img)
    mag = mag.flatten()
    theta = rbf_map(theta.flatten(), len(mag), sigma=sigma)
    mag = np.sqrt(1.0/len(mag)*2) * np.matmul(orf_matrix((len(mag)*2,len(mag))), mag)
    
    position = rbf_map(z, D/2)
    grad = np.matmul(orf_matrix((k,len(mag))),mag*theta)
    
    return np.kron(grad, position)

    
def kgrad(X, Y):
    """
    Computes kgrad using descriptor estimated
    """
    return np.matmul(kdes_grad(X), kdes_grad(Y))
    
def gradient(img, eps=1e-5):
    """
    Computes gradient of image after gaussian smoothing, similar to SIFT filters
    
    Follows example from https://gist.github.com/hackintoshrao/cc0e10f58079ede6866111d6ed75b420
    https://github.com/flyfj/RGBDMaze/blob/master/KernelDescriptor/kdes/gradkdes_dense.m
    
    img: 2d array of an image
    
    Return:
    mag: array of magnitude
    theta: array of orientation
    """
    sobelx = cv2.Sobel(img,-1,1,0,ksize=3)
    sobely = cv2.Sobel(img,-1,0,1,ksize=3) 
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    t = np.arctan2(sobely, sobelx)
    #normalize
    mag = mag / np.sqrt(np.sum(mag ** 2))
    
    return mag, t

def orf_matrix(size, sigma=1):
    """
    Generates SQ where S diagonal matrix with entries sampled from chi-dist with d-degree freedom and
    Orthogonal random gaussian matrix Q with each entry sampled independently from N(0,1)
    
    size: tuple (D,d)
    
    Return:
    SQ: ndarray where:
        S: ndarray diagonal sampled from chi d-degree
        Q: ndarray random gaussian
    """
    D,d = size
    if D <= d:
        S = np.diag(np.random.chisquare(d, size=D))
        G = np.random.normal(size=(d,d))
        Q, R = np.linalg.qr(G)
        return np.matmul(S,Q[:D])
    
    #stacking ind. G if D > d
    else:
        S = np.diag(np.random.chisquare(d, size=D))
        G_stack = [np.random.normal(size=(d,d)) for i in range(D/d+1)]
        Q_stack = np.concatenate([np.linalg.qr(G_stack[i])[0] for i in range(len(G_stack))], axis=0)[:D]
        return (1.0/sigma) * np.matmul(S, Q_stack)
    
def rbf_map(x, D, sigma=1):
    """
    RBF random feature map approximated by random gaussian matrix
    phi(x) = sqrt(1/D)[sin(w1T x),..., sin(wDT x), cos(w1T x),..., cos(wDT x)]T
    
    x: array
    D: int desired dimension
    Return:
    phi_x: array of mapped feature
    """
    d = len(x)
    W = orf_matrix((D,d), sigma=sigma)

    #compute sin and cos
    phi_x = np.zeros(2*D)
    for i in range(D):
        phi_x[i] = np.sin(np.matmul(W[i], x))
        phi_x[i+D] = np.cos(np.matmul(W[i], x))
    return np.sqrt(1.0/D) * phi_x

def ang_map(x, D):
    """
    Angular kernel feature map with random Gaussian
    
    x: array
    D: int mapped dimension
    
    Return:
    phi_x: array of mapped feature
    """
    d = len(x)
    G = np.random.normal(size=(D,d))
    return 1.0/np.sqrt(D) * np.sign(np.matmul(G,x))
    