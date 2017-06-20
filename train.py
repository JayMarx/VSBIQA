# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:49:24 2017

@author: justjay
"""
import numpy as np
import random
import argparse
from iterator import SerialIterator

import chainer
from chainer import cuda
from chainer import serializers
from chainer import optimizers
from chainer import datasets
from chainer import iterators
from chainer import training 
from chainer.training import extensions

import cv2
from sklearn.feature_extraction.image import extract_patches

###############修改
#from nr_model import Model
from model import Model

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('--model', '-m', default='', 
                    help='path to the trained model')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID')

args = parser.parse_args()


###########修改
#model = Model('patchwise')
model = Model()

cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
model.to_gpu()
#serializers.load_hdf5(args.model, model)

train_pair_path = 'data/Annotations/train_label.txt'
image_path = 'data/Imageset/live2_renamed_images'
save_model_path = 'models/nr_jay_live2.model'
file_path = './patches_weight.txt'

#get the train dataset:(image_array, label), in the order of label
train = datasets.LabeledImageDataset(train_pair_path, image_path, dtype=np.float32, label_dtype=np.int32)
#val = datasets.LabeledImageDataset(val_pair_path, image_path, dtype=np.float32, label_dtype=np.int32)

train_img_num = len(train)
#val_img_num = len(val)

#print img_num
patches_per_img = 256
#extract all the images to 32x32 pixls

f_label = open(train_pair_path, 'r')
f_patch = open(file_path, 'r')

list_label = [line.split()[0] for line in f_label.readlines()]          #get names of training images
list_patch_img = [line.split()[0] for line in f_patch.readlines()]      #get names of image patches
f_patch.seek(0)                                                         #reset file pointer       
list_patch = [line.split() for line in f_patch.readlines()]             #get patches info

f_label.close()
f_patch.close()

print '-------------Load data-------------'
train_patches = []
for i in range(train_img_num):
    patches = extract_patches(train[i][0], (3,32,32), 32)
    temp = patches.reshape((-1, 3, 32, 32))
    
    #select 128 top patches/image
    line_index = list_patch_img.index(list_label[i])                    #find the location of corresponding patches
    #print i, line_index, list_patch_img[line_index], len(temp)
        
    temp_slice = [temp[int(index)] for index in list_patch[line_index][1:patches_per_img+1]]
    
    #print temp_slice    
    #temp_slice = random.sample(temp, patches_per_img)    
    for j in range(len(temp_slice)):
        temp_slice[j] = xp.array(temp_slice[j].astype(np.float32))      
        train_patches.append((temp_slice[j], train[i][1]))

print '--------------Done!----------------'
        
#use my own SerialIterator, pick image patches randomly
train_iter = SerialIterator(train_patches, patches = patches_per_img, batch_size=4)
#val_iter = SerialIterator(val_patches, patches = patches_per_img, batch_size=4, repeat=False, shuffle=False)
optimizer = optimizers.Adam()
#optimizer = optimizers.MomentumSGD()
optimizer.use_cleargrads()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (8000, 'epoch'), out='result')

trainer.extend(extensions.LogReport())

trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'elapsed_time']))

#trainer.extend(extensions.ProgressBar())

trainer.run()
serializers.save_hdf5(save_model_path, model)


    
