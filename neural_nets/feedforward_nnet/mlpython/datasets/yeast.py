# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

"""
Module ``datasets.yeast`` gives access to the Yeast dataset.

| **Reference:** 
| A kernel method for multi-labelled classification
| Elisseeff and Weston
| http://books.nips.cc/papers/files/nips14/AA45.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the Yeast dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'target_size'``
    * ``'length'``

    """
    
    input_size=103
    target_size=14
    dir_path = os.path.expanduser(dir_path)
    def convert_target(target_str):
        targets = np.zeros((target_size))
        for l in target_str.split(','):
            id = int(l)
            targets[id] = 1
        return targets

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=False,input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'yeast_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [1250,250,917]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(target_size,)],[np.float64,bool],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,'target_size':target_size,
                                        'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_train.svm.bz2',os.path.join(dir_path,'yeast_train.svm.bz2'))
    urllib.urlretrieve('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_test.svm.bz2',os.path.join(dir_path,'yeast_test.svm.bz2'))

    import bz2
    train_valid_bz2_file = bz2.BZ2File(os.path.join(dir_path,'yeast_train.svm.bz2'))
    test_bz2_file = bz2.BZ2File(os.path.join(dir_path,'yeast_test.svm.bz2'))

    print 'Splitting training set into smaller training/validation sets'
    train_file,valid_file,test_file = [open(os.path.join(dir_path, 'yeast_' + ds + '.libsvm'),'w') for ds in ['train','valid','test']]

    # Putting train/valid data in memory
    train_valid_data = [ line for line in train_valid_bz2_file ]

    # Shuffle data
    import random
    random.seed(12345)
    perm = range(len(train_valid_data))
    random.shuffle(perm)
    line_id = 0
    train_valid_split = 1250
    for i in perm:
        s = train_valid_data[i]
        if line_id < train_valid_split:
            train_file.write(s)
        else:
            valid_file.write(s)
        line_id += 1
    train_file.close()
    valid_file.close()
    train_valid_bz2_file.close()

    for line in test_bz2_file:
        test_file.write(line)

    test_file.close()
    test_bz2_file.close()
    print 'Done                     '
