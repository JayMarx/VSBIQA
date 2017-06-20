#modified by jieli
from __future__ import division

import numpy

from chainer.dataset import iterator
import cv2

class SerialIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

    """

    def __init__(self, dataset, patches, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self.patches = patches
        #dataset_patches_len = len(dataset)
        dataset_patches_len = int((len(dataset))/(self.patches)) #get the num of images,len(dataset) get the num of all patches
        #print dataset_patches_len        
        if shuffle:
            #self._order = numpy.random.permutation(len(dataset))
            self._order = numpy.random.permutation(dataset_patches_len)
            #print self._order
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        #print 'i:', i
        #print 'i_end:', i_end
        #N = len(self.dataset)
        N = int((len(self.dataset))/(self.patches))
        
        if self._order is None:
            #batch = self.dataset[i:i_end]
            batch = [self.dataset[i*(self.patches):i_end*(self.patches)]]
        else:
            #batch = [self.dataset[index] for index in self._order[i:i_end]]
            batch = [self.dataset[index*(self.patches):(index+1)*(self.patches)] for index in self._order[i:i_end]]
            #print batch
        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch += list(self.dataset[:rest*(self.patches)])
                    else:
                        batch += [self.dataset[index*(self.patches):(index+1)*(self.patches)]
                                  for index in self._order[:rest]]
                self.current_position = rest
            else:
                self.current_position = N

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch[0]

        #return batch
    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
