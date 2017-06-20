# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:31:34 2017

@author: justjay
"""
#LIVE2 split images randomly 
#Train: 17, Validation: 6, Test: 6
#Train: 23, Test: 6; train:test ~= 8:2  

import numpy as np
import scipy.io as sio

names_mat = sio.loadmat('./refnames_all.mat')
dmos_mat = sio.loadmat('./dmos.mat')
dmos_dmos = dmos_mat['dmos']

all_imgs = names_mat['refnames_all']
index_jpg2k = np.arange(227)
index_jpg   = np.arange(227, 460)
index_wn    = np.arange(460, 634)
index_blur  = np.arange(634, 808)
index_ff    = np.arange(808, 982)

index = [index_jpg2k, index_jpg, index_wn, index_blur, index_ff]

names = ('coinsinfountain.bmp', 'ocean.bmp', 'statue.bmp', 'dancers.bmp',
         'paintedhouse.bmp', 'stream.bmp', 'bikes.bmp', 'flowersonih35.bmp', 
         'parrots.bmp', 'studentsculpture.bmp', 'building2.bmp', 'plane.bmp',
         'woman.bmp', 'buildings.bmp', 'house.bmp', 'rapids.bmp', 'womanhat.bmp',
         'caps.bmp', 'lighthouse.bmp', 'sailing1.bmp', 'carnivaldolls.bmp',
         'lighthouse2.bmp', 'sailing2.bmp', 'cemetry.bmp', 'manfishing.bmp',
         'sailing3.bmp', 'churchandcapitol.bmp', 'monarch.bmp', 'sailing4.bmp')

train_dir = './train_label.txt'
#val_dir = './val_label.txt'
test_dir = './test.txt'

train_ptr = open(train_dir, 'w')
#val_ptr = open(val_dir, 'w')
test_ptr = open(test_dir, 'w')

#整体训练、测试
for m in range(5):
    #分别对图片ID按names排序，分29组
    imgbynames = [[] for i in range(len(names))]
    for j in range(len(names)):
        for i in index[m]:
            if all_imgs[0][i][0] == names[j]:
                imgbynames[j].append(i)
    
    new_order = np.random.permutation(len(names))
    
    #write train images,without validation: 17->23
    for i in new_order[0:23]:
        for j in range(len(imgbynames[i])):
            train_ptr.write('{:06d}.bmp {:d}\n'.format(int(imgbynames[i][j])+1, int(np.round(dmos_dmos[0][imgbynames[i][j]]))))
    
    #use validation
    #write validation images
#    for i in new_order[17:23]:
#        for j in range(len(imgbynames[i])):
#            val_ptr.write('{:06d}.bmp {:d}\n'.format(int(imgbynames[i][j])+1, int(np.round(dmos_dmos[0][imgbynames[i][j]]))))
    
    #write test images
    for i in new_order[23:29]:
        for j in range(len(imgbynames[i])):
            test_ptr.write('{:06d}.bmp\n'.format(int(imgbynames[i][j])+1))

#分类别训练、测试
#m = 4
##分别对图片ID按names排序，分29组
#imgbynames = [[] for i in range(len(names))]
#for j in range(len(names)):
#    for i in index[m]:
#        if all_imgs[0][i][0] == names[j]:
#            imgbynames[j].append(i)
#    
#    
##write train images,without validation: 17->23
#for i in new_order[0:23]:
#    for j in range(len(imgbynames[i])):
#        train_ptr.write('{:06d}.bmp {:d}\n'.format(int(imgbynames[i][j])+1, int(np.round(dmos_dmos[0][imgbynames[i][j]]))))
#    
##write test images
#for i in new_order[23:29]:
#    for j in range(len(imgbynames[i])):
#        test_ptr.write('{:06d}.bmp\n'.format(int(imgbynames[i][j])+1))

train_ptr.close()
#val_ptr.close()
test_ptr.close()
