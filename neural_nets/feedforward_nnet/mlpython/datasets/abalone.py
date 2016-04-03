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
Module ``datasets.abalone`` gives access to the Abalone dataset.

The Abalone dataset is obtained here: http://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/regression.html#abalone.

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the Abalone dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'length'``

    """
    
    input_size = 8
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        return mlio.libsvm_load_line(line, float, float, sparse=False, input_size=input_size)

    train_file,valid_file,test_file = [os.path.join(dir_path, 'abalone_' + ds + '.libsvm') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [3341, 418, 418]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,np.float64],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size, 'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """
    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
 
    # Get the main file, will be used to create train, valid and test file.
    urllib.urlretrieve('http://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/regression/abalone', os.path.join(dir_path, 'abalone_temp.libsvm'))

    # Create files
    train_file = open(os.path.join(dir_path, 'abalone_train.libsvm'), "w")
    valid_file = open(os.path.join(dir_path, 'abalone_valid.libsvm'), "w")
    test_file = open(os.path.join(dir_path, 'abalone_test.libsvm'), "w")
    
    # Split 80%, 10%, 10% (train,valid,test)
    fp = open(os.path.join(dir_path, 'abalone_temp.libsvm'))
  
    # Add the lines of the file into a list
    lineList = []
    for line in fp:
        lineList.append(line)
    
    # Shuffle
    import random
    random.seed(25)
    random.shuffle(lineList)
    
    # Write lines into each file
    for i, line in enumerate(lineList):
        if i < 3341:
            train_file.write(line)
        elif i < 3759:
            valid_file.write(line)
        else:
            test_file.write(line)
    fp.close()

    train_file.close()
    valid_file.close()
    test_file.close()
    
    # Delete Temp file
    os.remove(os.path.join(dir_path,'abalone_temp.libsvm'))
    
    print 'Done'
