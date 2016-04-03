# Copyright 2011 Guillaume Roy-Fontaine and David Brouillard. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY Guillaume Roy-Fontaine and David Brouillard ``AS IS'' AND ANY EXPRESS OR IMPLIED
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
# or implied, of Guillaume Roy-Fontaine and David Brouillard.

"""
Module ``datasets.heart`` gives access to the Heart (SPECT) dataset.

The Heart dataset is obtained here: http://archive.ics.uci.edu/ml/machine-learning-databases/spect.

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the Heart dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size=22
    targets = set(range(2))
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:-1]]),int(tokens[-1]))
        #return mlio.libsvm_load_line(line,float,int,sparse,input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'heart_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [50,30,187]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l,'targets':targets} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """
    
    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train',os.path.join(dir_path,'heart.train'))
    urllib.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test',os.path.join(dir_path,'heart.test'))
    print 'Splitting dataset into training/validation/test sets'
    file_train_and_valid = open(os.path.join(dir_path,'heart.train'))
    file_test = open(os.path.join(dir_path,'heart.test'))
    train_file,valid_file,test_file = [open(os.path.join(dir_path, 'heart_' + ds + '.txt'),'w') for ds in ['train','valid','test']]

    # Putting all data in memory
    train_and_valid_data = []
    for line in file_train_and_valid:
        tokens = line.strip('\n').strip(',').split(',')
        s = ''
        for t in range(1,len(tokens)):
            s = s + tokens[t] + ' '
        target = tokens[0]
        s = s + str(target) + '\n'
        train_and_valid_data += [s]

    for line in file_test:
        tokens = line.strip('\n').strip(',').split(',')
        s = ''
        for t in range(1,len(tokens)):
            s = s + tokens[t] + ' '
        target = tokens[0]
        s = s + str(target) + '\n'
        test_file.write(s)
    test_file.close()

    # Shuffle data
    import random
    random.seed(25)
    perm = range(len(train_and_valid_data))
    random.shuffle(perm)
    line_id = 0
    train_valid_split = 50
    for i in perm:
        s = train_and_valid_data[i]
        if line_id < train_valid_split:
            train_file.write(s)
        else:
            valid_file.write(s)
        line_id += 1
    train_file.close()
    valid_file.close()
    print 'Done                     '
