#! /usr/bin/env python

# Copyright 2014 Frederic Bergeron & Benoit Gauthier. All rights reserved.
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

import mlpython.datasets.store as dataset_store
import cPickle,sys
import numpy as np
import sys
import os
from pdb import set_trace as dbg

def save(p, f):
	"""
	Pickles object ``p`` and saves it to file ``f``.
	The file must be open before that call
	and close after it.
	"""
	cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 

def savefirstandlast(myIterator,myFile, numbertosave =10):
    listToSave = list()
    for i in range(numbertosave):
        listToSave.append(myIterator.next())
    for value in listToSave:
        save(value, myFile)
    for value in myIterator:
        listToSave.pop(0)
        listToSave.append(value)
    for value in listToSave:
        save(value, myFile)

#Running as a script
def main():
    datasetName = sys.argv[1]
    exec 'import mlpython.datasets.'+datasetName+' as mldataset'

    f =file('./unit_tests/test_datasets/pickles/'+datasetName+'.pkl','wb')
    dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + datasetName
    dictionnary = mldataset.load(dataset_dir, False)
    train = dictionnary['train']
    valid = dictionnary['valid']
    test = dictionnary['test']
    
    savefirstandlast(iter(train[0]),f)
    savefirstandlast(iter(valid[0]),f)
    savefirstandlast(iter(test[0]),f)

    '''Adding all the metadata to the file.'''
    save(train[1],f)
    save(valid[1],f)
    save(test[1],f)
    f.close()

if __name__ == "__main__":
    main()