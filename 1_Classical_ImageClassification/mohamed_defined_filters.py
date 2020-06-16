import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import laplace, sobel, roberts, gaussian,apply_hysteresis_threshold, threshold_yen, threshold_mean
from skimage.feature import canny, hog
from skimage.util import invert
from skimage.morphology import skeletonize
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral,denoise_wavelet, estimate_sigma

#from skimage.filters import sobel

def hyst_tresh_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                hyst_treshold = apply_hysteresis_threshold(img, 1.5, 2.5)
                
                features.append(hyst_treshold.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames



def laplace_filt (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplace_filter = laplace(img, ksize=3, mask=None)
                
                features.append(laplace_filter.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def sobel_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sobel_filt = sobel(img)
                
                features.append(sobel_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def roberts_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roberts_filt = roberts(img)
                
                features.append(roberts_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def gaussian_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gaussian_filt = gaussian(img, sigma=0.6)
                
                features.append(gaussian_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def threshold_yen_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                threshold_yen_filt = threshold_yen(img)
                
                features.append(threshold_yen_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def threshold_mean_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                threshold_mean_filt = threshold_mean(img)
                
                features.append(threshold_mean_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def canny_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                canny_filt = canny(img, sigma = 0.5)
                
                features.append(canny_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames


def skeletonize_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = invert(img)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                skeletonize_filt = skeletonize(img)
                
                features.append(skeletonize_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def denoise_tv_chambolle_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                
                denoise_tv_chambolle_filt = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
                
                features.append(denoise_tv_chambolle_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames


def denoise_wavelet_filter (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                
                denoise_wavelet_filt = denoise_wavelet(img, multichannel=True, rescale_sigma=True)
                
                features.append(denoise_wavelet_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames

def denoise_and_hog (folder): # iterate through folders, assembling feature, label, and classname data objects
    class_id = 0
    features = []
    labels = np.array([])
    classnames = []
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            #print("Reading data from", d)
            classnames.append(d) # use the folder name as the class name for this label
            files = os.listdir(os.path.join(root,d))
            for f in files:
                imgFile = os.path.join(root,d, f) # Load the image file
                img = plt.imread(imgFile)
                img = cv2.resize(img, (128, 128)) # Resizing all the images to insure proper reading
                
                denoise_filt = denoise_wavelet(img, multichannel=True, rescale_sigma=True)
                
                hog_des = hog(denoise_filt, block_norm='L2-Hys', pixels_per_cell=(9, 9),
                                          cells_per_block=(3, 3), orientations=6,  transform_sqrt=True)

                features.append(hog_des.ravel())
                #features.append(sobel_filt.ravel())
                labels = np.append(labels, class_id ) # Add it to the numpy array of labels
            class_id  += 1
            
    features = np.array(features) # Convert the list of features into a numpy array
    return features, labels, classnames
