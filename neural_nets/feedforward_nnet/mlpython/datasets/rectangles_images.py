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
Module ``datasets.rectangles_images`` gives access to the Rectangles images dataset.

| **Reference:** 
| An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation
| Larochelle, Erhan, Courville, Bergstra and Bengio
| http://www.dmi.usherb.ca/~larocheh/publications/deep-nets-icml-07.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the Rectangles images dataset. 

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size = 784
    dir_path = os.path.expanduser(dir_path)
    targets = set(range(2))
        
    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:-1]]), float(tokens[-1]))
        


    train_file,valid_file,test_file = [os.path.join(dir_path, 'rectangles_images_' + ds + '.amat') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [10000, 2000, 50000]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]  
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size, 'length':l, 'targets':targets} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib

    ## Download the main zip file
    urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip',os.path.join(dir_path,'rectangles_images.zip'))
    
    # Extract the zip file
    print 'Extracting the dataset'
    import zipfile
    fh = open(os.path.join(dir_path,'rectangles_images.zip'), 'rb')
    z = zipfile.ZipFile(fh)
    for name in z.namelist():
        s = name.split('/')
        outfile = open(os.path.join(dir_path, s[len(s) -1]), 'wb')
        outfile.write(z.read(name))
        outfile.close()
    fh.close()
        
    train_file_path = os.path.join(dir_path,'rectangles_images_train.amat')
    valid_file_path = os.path.join(dir_path,'rectangles_images_valid.amat')
    test_file_path = os.path.join(dir_path,'rectangles_images_test.amat')

    # Rename train and test files
    os.rename(os.path.join(dir_path,'rectangles_im_train.amat'), train_file_path)
    os.rename(os.path.join(dir_path,'rectangles_im_test.amat'), test_file_path)    

    # Split data in valid file and train file
    fp = open(train_file_path)
  
    # Add the lines of the file into a list
    lineList = []
    for line in fp:
        lineList.append(line)
    fp.close()
        
    # Create valid file and train file
    valid_file = open(valid_file_path, "w")
    train_file = open(train_file_path, "w")
    
    # Write lines into valid file and train file
    for i, line in enumerate(lineList):
        if ((i + 1) > 10000):
            valid_file.write(line)
        else:
            train_file.write(line)
    
    valid_file.close()
    train_file.close()
    
    # Delete Temp file
    os.remove(os.path.join(dir_path,'rectangles_images.zip'))
    
    print 'Done                     '

