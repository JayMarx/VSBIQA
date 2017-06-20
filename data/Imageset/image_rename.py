# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:55:23 2017

@author: justjay
"""

#对live2图片重命名

import os

namelist = os.listdir('./fastfading/')
for name in namelist:
    if name.endswith('.bmp'):
        oldname = name
        newname = '%06d' % (int(oldname[3:-4])) + '.bmp'
        os.rename('./fastfading/'+oldname, './fastfading/'+newname)
        