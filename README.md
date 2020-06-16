# ComputerVision
In this repository, I share some projects I did in my Computer Vision class at IE HST as well as personal projects.

# 1. Classical Approach to Image Classification 

The following project focuses on Image Classification, a ML task of assigning the most likely label to input images for given categories. And here I build a classification pipeline to classify images of 3 categories: Automobiles, Planes, and Trains.

I am thus focusing on implementing feature-based classifiers rather than more advanced techniques such as CNN. The main goal is to assess the impact of preprocessing and feature extraction from images on ML classifiers,

The classifers assessed here include: `Decision Tree Classifier` , `Random Forrest`, `XGBClassifier`, `LGBMClassifier`, `Multi-layer Perceptron classifier`, `CatBoostClassifier`, `SVM`, `Naive Bayes`, and `SGD Classifier`.

The processing steps include: `Histogram of Oriented Gradients (HOG)`, `Hysteresis thresholding`, `Mean thresholding`, `Yen thresholding`, `Edge Detectors`, `Laplace Filter`, `Sobel Filter`, `Roberts Filter`, `Canny Filter`, `Skeletonize`, `Gaussian Filter`, `TV_chambolle Denoising`, and `Wavelet Denoising`. 

Some of the transformations mentionned can be seen here:
![Transformations](/1_Classical_ImageClassification/processing.png)
