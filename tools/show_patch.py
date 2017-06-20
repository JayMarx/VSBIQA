# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:19:42 2017
只显示选中的128块patches，其余添0
@author: cvpr
"""

import cv2
import numpy as np

patch_txt_path = '../patches_weight.txt'
img_path = '../data/Imageset/live2_renamed_images/000022.bmp'

img = cv2.imread(img_path)
f = open(patch_txt_path, 'rt')

line = f.readlines()[21].split()[1:]
line_int = [int(i) for i in line]
#patch_img = np.zeros(img.shape)

height = img.shape[0]
width = img.shape[1]

row_num = width/32
column_num = height/32 

#保存感兴趣的patch
#mm = np.zeros((32,32,3))
#for i in range(row_num*column_num):
#    if i in line_int[:128]:
#        patch_row = i/row_num
#        patch_column = i-patch_row*row_num       
#        mm = img[patch_row*32:(patch_row+1)*32, patch_column*32:(patch_column+1)*32]
#        cv2.imwrite('./patches/{:d}.bmp'.format(i), mm)

#将不在top128内的patches置0
for i in range(row_num*column_num):
    if i not in line_int[:128]:
        patch_row = i/row_num
        patch_column = i-patch_row*row_num
        img[patch_row*32:(patch_row+1)*32, patch_column*32:(patch_column+1)*32] = 0


cv2.imwrite('./crop.bmp', img)

#cv2.imshow('pic', img)
#cv2.waitKey(0)

f.close()

