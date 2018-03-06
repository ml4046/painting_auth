import numpy as np
import pandas as pd
import imageio as im
import pywt
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob

from PIL import Image
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def decomp(x, level, wavelet='haar'):
    """
    Decomposes 2d array to a given level with wavelet
    Default padding: symmetric
    
    Return: cA, [[cH1, cV1, cD1],...,[cHn, cVn, cDn]]
    
                                -------------------
                                |        |        |
                                | cA(LL) | cH(LH) |
                                |        |        |
    (cA, (cH, cV, cD))  <--->   -------------------
                                |        |        |
                                | cV(HL) | cD(HH) |
                                |        |        |
                                -------------------
    """
    coeff = pywt.wavedec2(x, wavelet, level=level)
    return coeff[0], coeff[::-1]


def neigh_matrix(band, curr, next1, next2):
    """
    Returns matrix for linear_predictor [cH, cV, cD, cH1,...,cH2,...cD2]
    """
    try:
        if band == 'cH':
            subband = curr[0]
            rm_i = 0
        elif band == 'cV':
            subband = curr[1]
            rm_i = 1
        elif band == 'cD':
            subband = curr[2]
            rm_i = 2
    except:
        print 'Undefined Subband'
    
    Q = []
    
    #current subband neighbors
    for i in range(subband.shape[0]):
        for j in range(subband.shape[1]):
            neighs = []
            for sub in range(len(curr)):
                #appending current level neighbors
                if sub == rm_i:
                    #exclude current subband
                    neighs = np.append(neighs, get_neighbors(i, j, curr[sub], True))
                else:
                    neighs = np.append(neighs, get_neighbors(i, j, curr[sub]))
                #appending level +1, +2
                neighs = np.append(neighs, get_neighbors(i/2, j/2, next1[sub]))
                neighs = np.append(neighs, get_neighbors(i/4, j/4, next2[sub]))
                  
            Q.append(neighs)
                    
    return np.array(Q)

            

def get_neighbors(x, y, subband, exclude=False):
    """
    Returns all neighbors of subband at position x, y
    exclude: removes [0,0] if True
    """
    if len(subband.shape) != 2:
        return 'subband is < 2x2'

    window = [-1,0,1]
    bounds = [-1, subband.shape[0], subband.shape[1]]
    neighs = []
    
    for i in window:
        for j in window:
            if not ((x+i) in bounds) and not ((y+j) in bounds):
                neighs.append(subband[x+i][y+j])
            else:
                neighs.append(0)
                
    if exclude:
        neighs.pop(4)
            
    return np.array(neighs)
    

    
def linear_predictor(Q, subband):
    """
    Predictor for 3x3 neighborhood for a given subband level
    
    Param:
        subband: subband value
        Q: matrix containing all neighbors
        
    Return: 
        w: w weights for subbands \in {cH, cV, cD}
        res: residuals
    """

    w, s, r, res = np.linalg.lstsq(np.abs(Q), np.abs(subband).flatten())
    return w

def get_features(x, level, wavelet='haar'):
    """
    Extracts features for a given level of subbands
    
    Param:
        x: ndarray image data
        level: level to be generated at
    """
    features = []
    cA, coeff = decomp(x, level, wavelet=wavelet)
    
    for l in range(len(coeff)-3): #level
        for s in range(len(coeff[l])): #subband
            #generate statistics
            mean, var, skew, kurt = get_stats(coeff[l][s].flatten())
            features.extend((mean, var, skew, kurt))
            
            
    for l in range(len(coeff)-3):
        err = generate_error_stats(coeff[l], coeff[l+1], coeff[l+2])
        features.extend(err.flatten())
    
            
        
    return np.array(features)

def generate_features(imgs, y, level=6):
    """
    Generates features for a set of images
    
    Param:
        imgs: (K,M,N) K samples of M,N img matrix
    Return:
        features: (K,n) K samples of n features
    """
    features = []
    
    for i in range(len(imgs)):
        if len(imgs[i].shape) != 2:
            y = np.delete(y, i, 0)
        else:
            features.append(get_features(imgs[i], level=level))
        
    return (np.array(features), y)
    
def generate_error_stats(curr, next1, next2, eps=0.001):
    """
    Return error stats for given level based on neighborhood prediction
    Error based on log2 error
    """
    subbands = ['cH', 'cV', 'cD']
    error_stats = []
    
    for i in range(len(subbands)):
        Q = neigh_matrix(subbands[i], curr, next1, next2)
        w = linear_predictor(Q, curr[i])       
        #+1 to avoid 0 division error
        err = np.log2(abs(curr[i].flatten())+eps) - np.log2(abs(np.matmul(Q,w))+eps)
        mean, var, skew, kurtosis = get_stats(err)
        error_stats.append([mean, var, skew, kurtosis])
        
    return np.array(error_stats)
        

def get_stats(x):
    """
    Returns mean, var, skew, kurtosis of x
    """
    return x.mean(), x.var(), skew(x), kurtosis(x)
        

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def import_data(all_info, size=(256,256), label_name = 'artist', path='train_1/**', num_images=5000, show_step=100, all_im=False, one_hot=False):
    """
    Import images as grayscale, resizing to 256, 256
    Param:
        all_info: df data info
        label_name: str column name of desired label (can be genre)
        path: str path of data folder
        num_images: int number of images
        show_step: int print num. images imported every show_step
        all_im: bool import all images
        one_hot: bool return one_hot label instead of multiclass
    Return:
        images: array of images in grayscale
        mult: multi class labels
        one_hot: one hot labels
        
    """
    path = '/Users/Mason/Desktop/adv_big_data/' + path
    file_names = glob.glob(path)
    images = []
    names = []
    label = []
    
    if all_im:
        num_images = len(file_names)
    
    for i in range(num_images):
        if i % show_step == 0:
            print 'imported images: %d' % (i + 100)
        im_name = file_names[i].split('/')[-1]
        img = import_image(file_names[i],size=size)
        
        if len(img.shape) != 2:
            images.append(rgb2gray(img))
            label.append(all_info[all_info['filename']==im_name][label_name].values[0])
            names.append(im_name)
            
            
    #one-hot encode label
    lb = LabelBinarizer()
    one_hot = lb.fit_transform(label)
    
    #multiclass encode
    le = LabelEncoder()
    mult = le.fit_transform(label)
    
    return (np.array(images), mult, one_hot)


def import_image(path, size=(256,256)):
    """
    Imports an image with given path and resizes accordingly
    
    Params:
        path: str path of the image
        size: (m,n) desired size of the image
        
    Returns:
        img: (m,n) array of pixels
    """
    img = Image.open(path)
    if len(img.size) != 2:
        return np.array([])
    return np.asarray(img.resize(size))
    
    
def model(X, y):
    """
    Fit ML model to X given y labels
    """
    #model initialization
    model = svm.SVC(C=0.1)
    
    #train/test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    train_scores = cross_val_score(model, X, y, cv=20)
    
    print 'train scores..'
    print train_scores
    print train_scores.mean()
    return train_scores
    
    
    

if __name__ == "__main__":
    
    #import data info and images
    df = pd.read_csv('train_info.csv')
    imgs, label = import_data(df)
    
    #extract features
    X, y = generate_features(imgs, label)
    train_scores = model(X, y)