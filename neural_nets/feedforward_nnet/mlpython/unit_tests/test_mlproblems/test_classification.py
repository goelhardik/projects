# Copyright 2014 Frederic Bergeron. All rights reserved.
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
The ``test_mlproblems.test_classification`` module contains unit tests for MLProblems that are 
designed for classification problem.

This module contains the following classes:

* TestClassificationProblem:   Tests for a classification problem.
* TestClassSubsetProblem:      Tests for extraction of examples from a subset of all classes.

"""
from mlpython.mlproblems.classification import *
import numpy as np
from nose.tools import *

class TestClassificationProblem:

    @raises(KeyError)
    def test_missing_metadata(self):
        """Classification problem needs 'targets' metadata"""
        data = np.arange(10).reshape(5,2)
        cpb = ClassificationProblem(data)

    def test_class_to_id(self):
        """Classification problem creates 'class_to_id' metadata"""
        metadata = {'targets': ['yes','no']}
        data = np.arange(10).reshape(5,2)
        cpb = ClassificationProblem(data, metadata)

        results = {'yes': 0, 'no': 1}
        assert results == cpb.metadata['class_to_id']

    @raises(AttributeError)
    def test_iter_missing_metedata(self):
        """Classification problem needs 'class_to_id' metadata"""
        data = np.arange(10).reshape(5,2)
        cpb = ClassificationProblem(data,{},False)

        for item in cpb:
            #Should fail before this
            assert False

    def test_iter(self):
        """Classification problem iteration"""
        metadata = {'targets': ['a','c','e','g','i']}
        data = [[0,'a'],[2,'c'],[4,'e'],[6,'g'],[8,'i']]
        cpb = ClassificationProblem(data, metadata)

        results = [[0,0], [2,1],[4,2],[6,3],[8,4]]

        id = 0
        for input, target in cpb:
            assert input == results[id][0]
            assert target == results[id][1]
            id+=1

    def test_apply_on(self):
        """Classification problem apply_on passes 'class_to_id' metadata to the new problem"""
        metadata = {'targets': ['a','c','e','g','i']}
        data = [[0,'a'],[2,'c'],[4,'e'],[6,'g'],[8,'i']]
        cpb = ClassificationProblem(data, metadata)

        new_data = [[1,'a'],[3,'c'],[5,'e'],[7,'g'],[9,'i']]
        cpb2 = cpb.apply_on(new_data,{})

        results = {'a':0,'c':1,'e':2,'g':3,'i':4}
        assert cpb2.metadata['class_to_id'] == results



class TestClassSubsetProblem:

    def test_len_meta(self):
        """Classification subset problem uses 'class_subset_length' as data length"""

        metadata = {'targets': ['a','c','e','g','i']}
        data = [[0,'a'],[2,'c'],[4,'e'],[6,'g'],[8,'i']]
        cpb = ClassificationProblem(data, metadata)

        metadata = {'class_subset_length':20}

        cspb = ClassSubsetProblem(cpb, metadata)

        assert len(cspb) == 20

    def test_len(self):
        """Classification subset problem correctly count data in subset"""

        metadata = {'targets': ['a','c','e','g','i']}
        data = [[0,'a'],[2,'c'],[4,'e'],[6,'g'],[8,'i']]
        cpb = ClassificationProblem(data, metadata)

        subset = ['a', 'g']

        cspb = ClassSubsetProblem(cpb, {}, True, subset)
        assert len(cspb) == 2

    
