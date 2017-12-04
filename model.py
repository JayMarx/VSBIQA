# -*- coding: utf-8 -*-

"""
Created on Thu Mar  9 20:16:49 2017

@author: justjay
"""

import numpy as np

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter

cuda.check_cuda_available()
xp = cuda.cupy

class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(3, 32, 5),
            conv2 = L.Convolution2D(32, 64, 3),
            conv3 = L.Convolution2D(64, 128, 3),
 
            fc1   = L.Linear(512, 512),
            fc2   = L.Linear(512, 1),
            )

    def __call__(self, x_data, y_data, train=True, n_patches=32):
        if not isinstance(x_data, Variable):
            x = Variable(x_data, volatile=not train)
        else:
            x = x_data
            x_data = x.data
        
        ## self.n_images = y_data.shape[0]
        self.n_images = 1
        self.n_patches = x_data.shape[0]
        ## print x_data.shape[0]
        ## print y_data.shape[0]
        self.n_patches_per_image = self.n_patches / self.n_images
        ## self.n_patches_per_image = 32
      
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h,2)
        
        h_ = h
        self.h = h_

        ## h = F.dropout(F.relu(self.fc1(h_)), train=train, ratio=0.5)
        h = F.relu(self.fc1(h_))        
        h = self.fc2(h)
    
        a = xp.ones_like(h.data)
        ## y_data.data, to make y_data do not be a Variable
        t = xp.repeat(y_data.data, 1)
        t = xp.array(t.astype(np.float32))
        ## print h.data
        ## print len(t)            
        ## print t
        self.average_loss(h, a, t)


        if train:
            reporter.report({'loss': self.loss}, self)
            ## print 'self.lose:', self.loss.data
            return self.loss
        else:
            return self.loss, self.y

    def forward(self, x_data, y_data, train=True, n_patches=32):
        if not isinstance(x_data, Variable):
            x = Variable(x_data, volatile=not train)
        else:
            x = x_data
            x_data = x.data
            
        self.n_images = y_data.shape[0]
        self.n_patches = x_data.shape[0]
        self.n_patches_per_image = self.n_patches / self.n_images

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h,2)
        
        h_ = h
        self.h = h_

        ## h = F.dropout(F.relu(self.fc1(h_)), train=train, ratio=0.5)
        h = F.relu(self.fc1(h_))
        h = self.fc2(h)
        
        a = Variable(xp.ones_like(h.data), volatile=not train)
        t = Variable(xp.repeat(y_data, n_patches), volatile=not train)
        self.average_loss(h, a, t)

        if train:
            return self.loss
        else:
            return self.loss, self.y

    def average_loss(self, h, a, t):
        ## print F.reshape(t, (-1, 1)).data        
        ## print (h-F.reshape(t, (-1, 1))).data
        self.loss = F.sum(abs(h - F.reshape(t, (-1,1)))) 
        ## self.loss = F.sqrt(F.sum(F.square(h - F.reshape(t, (-1,1))))) 
        self.loss /= self.n_patches
        if self.n_images > 1:
            h = F.split_axis(h, self.n_images, 0)
            a = F.split_axis(a, self.n_images, 0)
        else:
            h, a = [h], [a]

    	self.y = h
    	self.a = a