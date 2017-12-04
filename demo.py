# -*- coding: utf-8 -*-

"""
Created on Fri Mar  3 11:04:15 2017

@author: justjay
"""
## Referring to the evaluate.py(dmaniry's deepIQA), modified by JayLee
## To evaluate all the images from the path in txt file, and save results

import os
import time
import numpy as np
import argparse

import chainer
import six
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import cv2
from PIL import Image
from sklearn.feature_extraction.image import extract_patches
    
from model import Model

parser = argparse.ArgumentParser(description='evaluate.py')
parser.add_argument('--model', '-m', default='./models/nr_jay_live2.model',
                    help='path to the trained model')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')

args = parser.parse_args()

patches_per_img = 256               ## test patches per image

model = Model()

cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(args.model, model)
## serializers.load_hdf5('./models/nr_jay_live2.model', model)

model.to_gpu()

## prepare data for LIVE2 database
test_label_path = 'data/Annotations/LIVE2/test.txt'
test_img_path = 'data/Imageset/live2/'
prewitt_img_path = 'data/Imageset/prewitt_images/'              ## path to gradient image
saliency_img_path = 'data/Imageset/saliency_images/'            ## path to saliency image

result_ptr = open('result/live2_result.txt', 'wt')
with open(test_label_path, 'rt') as f:
    for line in f:
        ## line = line.strip()                                  ## get test image name 
        line,la = line.strip().split()                          ## for debug
            
        tic = time.time()        
        full_path = os.path.join(test_img_path, line)
        ## img = cv2.imread(full_path)                          ## opencv读取图片矩阵格式与训练时读取的不一样
        ## 对应train时候的图片矩阵格式，使用Image函数读取，并transpose
        f = Image.open(full_path)        
        img = np.asarray(f, dtype=np.float32)
        img = img.transpose(2, 0, 1)
               
        ## img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = extract_patches(img, (3,32,32), 32)
    
        X = patches.reshape((-1, 3, 32, 32))                    ## X.shape:(patches_num, 3, 32, 32)
        y = []
        ## weights = []
        
        t = xp.zeros((1, 1), np.float32)
        ## t[0][0] = int(la)                                    ## for debug        
        
        patch_gradient = []
        patch_saliency = []
        patches_weight = []                                     ## weight patches
        weight_same = []                                        ## all weight is 1
        prewitt_img = cv2.imread(prewitt_img_path + line, 0)
        saliency_img = cv2.imread(saliency_img_path + line[0:-4] + '_HC.png', 0)
        height = prewitt_img.shape[0]
        width = prewitt_img.shape[1]
    
        ## calculate weight
        for m in range(height/32):
            for n in range(width/32):
                prewitt_img_crop = prewitt_img[m*32:(m+1)*32, n*32:(n+1)*32]
                saliency_img_crop = saliency_img[m*32:(m+1)*32, n*32:(n+1)*32]
                
                prewitt_sum = np.sum(prewitt_img_crop)/255
                patch_gradient.append(prewitt_sum)            
                
                saliency_sum = np.sum(saliency_img_crop)/255
                patch_saliency.append(saliency_sum)
                
                
                weight = [prewitt_sum*0.6+saliency_sum*0.4, m*(width/32)+n]             ## [weight, number of patches]
                patches_weight.append(weight)
        
        patches_weight.sort(reverse = True)        
    
        ## print patches_weight   
        
        index_order = [patches_weight[ind][1] for ind in range(len(patches_weight))]    ## 排序后各个patch在原图片的位置
        weight_order = []                                                               ## 排序后各个patch的权重
        
        X_batch = np.zeros(X.shape)
        for i in range(len(index_order)):
            X_batch[i] = X[index_order[i]] 
            weight_order.append(patches_weight[i][0].reshape((-1,)))
        
        ## X_batch = X[:X.shape[0]]
        X_batch = X_batch[:patches_per_img]
        X_batch = xp.array(X_batch.astype(np.float32))
        loss = model.forward(X_batch, t, False, X_batch.shape[0])
        
        ## print loss[0].data                                                           ## for debug            
        y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
        ## weight_same.append(xp.asnumpy(model.a[0].data).reshape((-1,)))
        
        print '%s    %f' % (line, time.time()-tic)
        
        y = np.concatenate(y)
        patches_weight_norm = np.concatenate(weight_order[:patches_per_img])
        ## patches_weight_norm = np.concatenate(weight_same[:128])
        
        score = np.abs(np.sum(y*patches_weight_norm)/np.sum(patches_weight_norm))
        result_ptr.write('{:s} {:f}\n'.format(line, score))

result_ptr.close()
## print("%f" %  (np.sum(y*weights)/np.sum(weights)))