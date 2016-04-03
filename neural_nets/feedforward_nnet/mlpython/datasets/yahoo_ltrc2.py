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
Module ``datasets.yahoo_ltrc`` gives access to Set 2 of the Yahoo!
Learning to Rank Challenge data. The queries correspond to query IDs,
while the inputs already contain query-dependent information.

| **Reference:** 
| Yahoo! Learning to Rank Challenge Overview
| Chapelle and Chang
| http://jmlr.csail.mit.edu/proceedings/papers/v14/chapelle11a/chapelle11a.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False,home_made_valid_split=False):
    """
    Loads the Yahoo! Learning to Rank Challenge, Set 2 data.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    Option ``home_made_valid_split`` determines whether the original
    training set should be further split into a "home made"
    train/valid split (default: False). If True, the dictionary mapping
    will contain 4 keys instead of 3: ``'train'`` (home made training set), 
    ``'valid'`` (home made validation set), ``'test'`` (original validation set)
    and ``'test2'`` (original test set).

    **Defined metadata:**

    * ``'input_size'``
    * ``'scores'``
    * ``'n_queries'``
    * ``'n_pairs'``
    * ``'length'``

    """
    
    input_size=700
    dir_path = os.path.expanduser(dir_path)
    sparse=False

    def convert(feature,value):
        if feature != 'qid':
            raise ValueError('Unexpected feature')
        return int(value)

    def load_line(line):
        return mlio.libsvm_load_line(line,convert,int,sparse,input_size)

    if home_made_valid_split:
        n_queries = [1000,266,1266,3798]
        lengths = [27244,7571,34881,103174]

        train_file,valid_file,test_file,test2_file = [os.path.join(dir_path, 'set2.' + ds + '.txt') for ds in ['in_house_train','in_house_valid','valid','test']]
        # Get data
        train,valid,test,test2 = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file,test2_file]]

        if load_to_memory:
            train,valid,test,test2 = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],[np.float64,int,int],l) for d,l in zip([train,valid,test,test2],lengths)]

        # Get metadata
        train_meta,valid_meta,test_meta,test2_meta = [{'input_size':input_size,
                                                       'scores':range(5),
                                                       'n_queries':nq,
                                                       'length':l,
                                                       'n_pairs':l} for nq,l in zip(n_queries,lengths)]

        return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta),'test2':(test2,test2_meta)}
    else:
        n_queries = [1266,1266,3798]
        lengths = [34815,34881,103174]

        # Get data file paths
        train_file,valid_file,test_file = [os.path.join(dir_path, 'set2.' + ds + '.txt') for ds in ['train','valid','test']]
        # Get data
        train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
        if load_to_memory:
            train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],[np.float64,int,int],l) for d,l in zip([train,valid,test],lengths)]

        train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                            'scores':range(5),
                                            'n_queries':nq,
                                            'length':l,
                                            'n_pairs':l} for nq,l in zip(n_queries,lengths)]

        return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}


def obtain(dir_path):
    """
    This dataset must be downloaded manually first through the Yahoo! Webscope Program at:
        http://webscope.sandbox.yahoo.com/
    Then, this function should be called to generate the necessary preprocessing of the data.
    """

    dir_path = os.path.expanduser(dir_path)
    train_file = os.path.join(dir_path, 'set2.train.txt')
    try:
        file = open(train_file)
        n_queries = 0
        in_house_train_file = os.path.join(dir_path, 'set2.in_house_train.txt')
        in_house_valid_file = os.path.join(dir_path, 'set2.in_house_valid.txt')
        train_file = open(in_house_train_file,'w')
        valid_file = open(in_house_valid_file,'w')
	# qids in validation set (sorry for the ridiculously long line...)
        qids_valid = [31055,30835,30987,29968,29979,31017,31138,30943,30647,31166,30340,30501,30915,29959,30796,30498,30651,31032,30952,30479,30206,30438,29924,30198,30666,30511,30866,30010,30204,30066,30453,29929,30767,30510,30698,31109,30900,30320,30402,30006,30167,30807,30792,30089,30634,30800,29985,30979,30777,30081,30381,30468,30805,30986,30569,30371,29957,30733,29972,31118,31039,30969,30202,30957,30870,30980,30542,30379,30035,30995,30656,29952,30638,30173,30460,30482,30732,30795,30341,31064,30201,31139,31103,30424,30002,31034,31142,30041,30554,30580,30933,30553,30973,30706,30558,30205,30159,30084,30760,31128,30748,30110,30134,30855,30273,30753,30112,29988,30075,29949,31135,31160,30311,30825,30102,30179,30636,30484,30403,30938,30693,30421,31018,30644,30709,31060,31152,30687,30215,30434,30126,30439,29945,29938,30109,29925,30256,30183,30337,30404,31116,31170,30414,30921,30563,30768,30582,30149,30114,30493,30131,30548,30143,31182,30248,30059,29934,30517,30108,30894,30499,30810,30683,29935,29965,30436,30630,30083,31059,30394,30182,30141,30442,30146,30850,30852,30017,30858,30444,29937,31165,30420,30267,30358,30888,30409,30236,30746,30082,30393,30324,30056,30191,30172,30620,30841,30947,31137,30462,31084,30949,30595,31078,30509,30711,30828,30225,30723,30609,30161,30270,30031,30290,30604,30287,30285,30532,30195,30873,30367,30527,30295,30040,30169,30720,30705,30680,29969,30655,30070,30853,29942,30013,30742,30375,30233,30521,30475,30487,31037,30165,30775,30306,30932,30881,30343,30535,30092,30252,30739,30427,30648,30360,30310,31134,30958,30819,30281,30329,30513,30093,30534,30948,30068,30528,30047]
	qids_valid = set(qids_valid)
        print 'Seperating training into smaller training/validation sets'
        qid = 0
        for line in file:
            new_qid = int(line.split('qid:')[1].split(' ')[0])
            if qid != new_qid:
                print '...reading query %i\r' % new_qid,
            qid = new_qid
            if qid in qids_valid:
                valid_file.write(line)
            else:
                train_file.write(line)
        train_file.close()
        valid_file.close()

        print 'Done                     '
    except IOError:
        print 'The data must first be downloaded manually through the Yahoo! Webscope Program at:'
        print '   http://webscope.sandbox.yahoo.com/'
        print 'Once this is done, put the uncompressed data in ' + dir_path
        print 'and call this function again.'
