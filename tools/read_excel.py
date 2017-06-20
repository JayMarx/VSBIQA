# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:53:19 2017

@author: cvpr
"""

import xlrd

data = xlrd.open_workbook('./csiq.DMOS.xlsx')
try:
    sheet = data.sheet_by_name('all_by_distortion')
except:
    print 'No sheet named: all_by_distortion'
    exit(0)

sheet_len = len(sheet.col(3))-4

file_path = './csiq_label.txt'
f_ptr = open(file_path, 'wt')

for i in range(sheet_len):
    dst_type = sheet.col(4)[i+4].value
    image = sheet.col(5)[i+4].value
    dst_lev = sheet.col(6)[i+4].value
    dmos = sheet.col(11)[i+4].value
    
    if dst_type == 'noise':
        dst_type = 'AWGN'
    elif dst_type == 'jpeg' or dst_type == 'blur':
        dst_type = dst_type.upper()
    elif dst_type[:4] == 'jpeg':
        dst_type = 'jpeg2000'
    else:
        dst_type = dst_type
    
    #print image, dst_type, dst_lev, dmos
    if type(image) == float:
        image = str(int(image))
    img_name = image + '.' + dst_type + '.' + str(int(dst_lev)) + '.png'
    f_ptr.write('{:s} {:.6f}\n'.format(img_name, dmos))

f_ptr.close()   
        