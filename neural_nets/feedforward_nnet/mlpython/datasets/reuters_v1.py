# Copyright 2014 Hugo Larochelle. All rights reserved.
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
Module ``datasets.reuters_v1`` gives access to the Reuters Corpus Volume 1: English Language. 1996-08-20 to 1997-08-19

This dataset is a corpus of English texts for natural language processing.

The orginal corpus is a set of XML files. Here we strip all of the XML to keep only the text.

| **Reference:** 
| Reuters Corpora (RCV1, RCV2, TRC2)
| NIST
| http://trec.nist.gov/data/reuters/reuters.html

The obtain function uses nltk. Installation instruction can be found here:
| http://www.nltk.org/install.html

"""

import mlpython.misc.io as mlio
import numpy as np
import os
import nltk

def load(dir_path,load_to_memory=False):
    """
    Loads the reuters corpus volume 1 dataset. This is actually txt version of it.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**

    * ``'length'``

    """
    
    dir_path = os.path.expanduser(dir_path)
    xmlfolder = os.path.join(dir_path, 'xml')
    filenames = os.listdir(xmlfolder)
    
    def load_file(file):
        raw = nltk.clean_html(file)
        return nltk.word_tokenize(raw)

    lengths = [564753,121018,121020]

    #lengths = [174800,1533,1535]
    #lengths = [7156,1533,1535]
    #lengths = [939,201,202]
    if load_to_memory:
        filenames = os.listdir(xmlfolder)
        raw_files = [None]*len(filenames)

        for i, filename in enumerate(filenames):
            with open(os.path.join(xmlfolder,filename), 'r') as current_file:
                file = current_file.read()
                raw_files[i] = load_file(file)

        train = raw_files[:7156]
        valid = raw_files[7156:8689]
        test = raw_files[8689:]
    else:
        for i, filename in enumerate(filenames):
            filenames[i] = os.path.join(xmlfolder,filename)
        train = mlio.load_from_files(filenames[:lengths[0]], load_file)
        valid = mlio.load_from_files(filenames[7156:8689], load_file)
        test = mlio.load_from_files(filenames[lengths[1]:], load_file)

    #    train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    This dataset must be optained manually first through the NIST at:
        http://trec.nist.gov/data/reuters/reuters.html
    Then, this function should be called to generate the necessary preprocessing of the data.
    It uses function from the nltk project. Installation instruction can be found here:
        http://www.nltk.org/install.html
    """

    dir_path = os.path.expanduser(dir_path)
    xmlfolder = os.path.join(dir_path, 'xml')

    if not os.path.isdir(xmlfolder):
        print 'This dataset expects to find xml files in a subfolder named "xml" in the given "dir_path".'
        print 'The corpus contains 2 files that you must move or delete: "codes.zip" and "dtds.zip".'
        print 'If you have the original zips from the reuters corpus, run the following code in a terminal.'
        print 'This will unzip all xml files contained in your current folder. Make sure you are in your dataset directory before running.'
        print '  unzip "*.zip" -d ./xml'
        print 'This command should take some time to execute.'

