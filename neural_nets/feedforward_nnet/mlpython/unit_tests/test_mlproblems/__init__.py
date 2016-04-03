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


import mlpython.mlproblems.generic as mlpbgen
import mlpython.mlproblems.classification as mlpbclass

def test_mlproblem_combinations():
    """
    Test a combination of many different MLProblems.
    """

    raw_data = zip(range(6),['A','A','B','C','A','B'])
    metadata = {'length':6,'targets':['A','B','C'],'input_size':1}
    
    def features(example,metadata):
        metadata['input_size'] = 2
        return ((example[0],example[0]),example[1])

    pb1 = mlpbgen.MLProblem(raw_data, metadata)
    print 'pb1',pb1.metadata
    pb2 = mlpbgen.SubsetProblem(pb1,subset=set([1,3,5]))
    print 'pb2',pb2.metadata
    pb3 = mlpbgen.MergedProblem([pb2,pb1])
    print 'pb3',pb3.metadata
    pb4 = mlpbgen.PreprocessedProblem(pb3,preprocess=features)
    print 'pb4',pb4.metadata
    pb5 = mlpbclass.ClassificationProblem(pb4)
    print 'pb5',pb5.metadata
    pb6 = mlpbclass.ClassSubsetProblem(pb5,subset=set(['A','C']))
    print 'pb6',pb6.metadata
    pb7 = mlpbgen.SubsetFieldsProblem(pb6,fields=[0,0,1])
    print 'pb7',pb7.metadata

    final_data = [[(1,1),(1,1),0],
                  [(3,3),(3,3),1],
                  [(0,0),(0,0),0],
                  [(1,1),(1,1),0],
                  [(3,3),(3,3),1],
                  [(4,4),(4,4),0]]
    final_metadata = {'input_size': 2, 'targets': set(['A', 'C']), 'class_to_id': {'A': 0, 'C': 1}}

    for ex1,ex2 in zip(pb7,final_data):
        assert cmp(ex1,ex2) == 0
    print pb7.metadata,final_metadata
    assert cmp(pb7.metadata,final_metadata) == 0
    
    raw_data2 = zip(range(6,10),['C','B','A','C'])
    metadata2 = {'length':4,'targets':['A','B','C'],'input_size':1}
    
    pbtest = pb7.apply_on(raw_data2,metadata2)
    final_test_data = [[(6,6),(6,6),1],
                       [(8,8),(8,8),0],
                       [(9,9),(9,9),1]]
    final_test_metadata = {'input_size': 2, 'targets': set(['A', 'C']), 'class_to_id': {'A': 0, 'C': 1}}

    for ex1,ex2 in zip(pbtest,final_test_data):
        assert cmp(ex1,ex2) == 0
    assert cmp(pbtest.metadata,final_test_metadata) == 0
