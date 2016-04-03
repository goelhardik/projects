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
Module ``datasets.cifar10`` gives access to the CIFAR-10 dataset.

| **Reference:** 
| Learning multiple layers of features from tiny images
| Alex Krizhevsky
| http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the CIFAR-10 dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'length'``
    * ``'targets'``
    * ``'class_to_id'``

    """

    input_size=3072
    dir_path = os.path.expanduser(dir_path)
    if load_to_memory:
        batch1 = mlio.load(os.path.join(dir_path,'data_batch_1'))
        batch2 = mlio.load(os.path.join(dir_path,'data_batch_2'))
        batch3 = mlio.load(os.path.join(dir_path,'data_batch_3'))
        batch4 = mlio.load(os.path.join(dir_path,'data_batch_4'))
        batch5 = mlio.load(os.path.join(dir_path,'data_batch_5'))
        testbatch = mlio.load(os.path.join(dir_path,'test_batch'))

        train_data = np.vstack([batch1['data'],batch2['data'],batch3['data'],batch4['data']])
        train_labels = np.hstack([batch1['labels'],batch2['labels'],batch3['labels'],batch4['labels']])
        train = mlio.IteratorWithFields(np.hstack([train_data,train_labels.reshape(-1,1)]),((0,input_size),(input_size,input_size+1)))

        valid_data = batch5['data']
        valid_labels = np.array(batch5['labels'])
        valid = mlio.IteratorWithFields(np.hstack([valid_data,valid_labels.reshape(-1,1)]),((0,input_size),(input_size,input_size+1)))

        test_data = testbatch['data']
        test_labels = np.array(testbatch['labels'])
        test = mlio.IteratorWithFields(np.hstack([test_data,test_labels.reshape(-1,1)]),((0,input_size),(input_size,input_size+1)))

    else:
        def load_line(line):
            tokens = line.split()
            return (np.array([int(i) for i in tokens[:-1]]),int(tokens[-1]))

        train_file,valid_file,test_file = [os.path.join(dir_path, 'cifar-10-' + ds + '.txt') for ds in ['train','valid','test']]
        # Get data
        train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    # Get metadata
    lengths = [40000,10000,10000]
    other_meta = mlio.load(os.path.join(dir_path,'batches.meta'))
    label_names = other_meta['label_names']
    targets = set(label_names)
    class_to_id = {}
    for i,c in enumerate(label_names):
        class_to_id[c] = i
        
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                        'length':l,'targets':targets,
                                        'class_to_id':class_to_id} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',os.path.join(dir_path,'cifar-10-python.tar.gz'))
    print 'Extracting the dataset (this could take a while)'
    import tarfile
    tf = tarfile.open(os.path.join(dir_path,'cifar-10-python.tar.gz'))
    tf.extractall(dir_path)
    import shutil
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/data_batch_1'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/data_batch_2'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/data_batch_3'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/data_batch_4'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/data_batch_5'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/test_batch'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/readme.html'),dir_path)
    shutil.move(os.path.join(dir_path,'cifar-10-batches-py/batches.meta'),dir_path)
    os.rmdir(os.path.join(dir_path,'cifar-10-batches-py'))

    # Putting stuff in ascii files to enable not loading in memory
    batch1 = mlio.load(os.path.join(dir_path,'data_batch_1'))
    batch2 = mlio.load(os.path.join(dir_path,'data_batch_2'))
    batch3 = mlio.load(os.path.join(dir_path,'data_batch_3'))
    batch4 = mlio.load(os.path.join(dir_path,'data_batch_4'))
    batch5 = mlio.load(os.path.join(dir_path,'data_batch_5'))
    testbatch = mlio.load(os.path.join(dir_path,'test_batch'))
    train_data = np.vstack([batch1['data'],batch2['data'],batch3['data'],batch4['data']])
    train_labels = np.hstack([batch1['labels'],batch2['labels'],batch3['labels'],batch4['labels']])
    valid_data = batch5['data']
    valid_labels = batch5['labels']
    test_data = testbatch['data']
    test_labels = testbatch['labels']

    def write_to_file(data,labels,file):
        f = open(os.path.join(dir_path,file),'w')
        for input,label in zip(data,labels):
            f.write(' '.join([str(xi) for xi in input]) + ' ' + str(label) + '\n')
        f.close()

    write_to_file(train_data,train_labels,'cifar-10-train.txt')
    write_to_file(valid_data,valid_labels,'cifar-10-valid.txt')
    write_to_file(test_data,test_labels,'cifar-10-test.txt')

    print 'Done                     '
