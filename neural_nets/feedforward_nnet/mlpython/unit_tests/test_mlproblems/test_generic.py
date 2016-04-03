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
The ``test_mlproblems.test_generic`` module contains unit tests for MLProblems that are not
designed for a specific type of problem.

This module contains the following classes:

* TestMLProblem:              Root test class for machine learning problems.
* TestSubsetProblem:          Tests extraction a subset of examples from a dataset.
* TestSubsetFieldsProblem:    Tests extraction a subset of the fields in a dataset.
* TestMergedProblem:          Tests the merge of several datasets together.
* TestPreprocessedProblem:    Tests the application of an arbitrary preprocessing on a dataset.
* TestSemisupervisedProblem:  Tests if the SemisupervisedProblem class removed the labels of a subset of the examples in a dataset.

"""
from mlpython.mlproblems.generic import *
import numpy as np

class TestMLProblem:

    def test_len(self):
        """MLProblem length is data length"""
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3}
        mlpb = MLProblem(data,metadata)
        assert len(mlpb) == 10

    def test_len_metadata(self):
        """MLProblem length is given length"""
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3, 'length':9}
        mlpb = MLProblem(data,metadata)
        assert len(mlpb) == 9
        #This is a lie that should never be done in a non-test context
        #Given length should always be the real length.

    def test_iter(self):
        """MLProblem iteration"""
        data = np.arange(30).reshape((10,3))
        mlpb = MLProblem(data)
        line = 0.
        for example in mlpb:
            array = np.array([line, line+1, line+2])
            assert np.array_equal(array, example)
            line += 3

    def test_peak(self):
        """MLProblem peak returns the first data"""
        data = np.arange(20).reshape((4,5))
        mlpb = MLProblem(data)
        peakLine = np.array([0,1,2,3,4])
        assert np.array_equal(peakLine, mlpb.peak())

    def test_apply_on(self):
        """MLProblem apply_on doesn't pass metadata"""
        data = np.arange(30).reshape((10,3))
        metadata = {'input_size':3, 'length':9}
        mlpb = MLProblem(data,metadata)

        new_data = np.arange(20).reshape((4,5))
        new_metadata = {'input_size':5, 'length':4}
        mlpb2 = MLProblem(new_data, new_metadata)

        assert len(mlpb2) == 4
        assert len(mlpb) == 9

        mlpb3 = mlpb.apply_on(mlpb2)
        assert len(mlpb3) == 4
        assert mlpb3.metadata['input_size'] == 5




class TestSubsetProblem:

    def test_len(self):
        """SubestProblem length"""
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,5])
        subpb = SubsetProblem(data,{},True, subset)

        assert len(subpb) == 3

    def test_iter(self):
        """SubsetProblem iteration"""
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,4])
        subpb = SubsetProblem(data,{},True, subset)

        results = np.array((0,1,2,3,4,5,12,13,14)).reshape(3,3)
        i = 0
        for line in subpb:
            assert np.array_equal(line, results[i])
            i+=1

    def test_apply_on(self):
        """SubsetProblem apply_on"""
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,4])
        subpb = SubsetProblem(data,{},True, subset)

        new_data = np.arange(10).reshape((5,2))
        fullSized = subpb.apply_on(new_data)
        assert len(fullSized) == 5

    def test_apply_on_parents(self):
        """SubsetProblem apply_on in chain"""
        data = np.arange(30).reshape((10,3))
        subset = set([0,1,4])
        mlpb = MLProblem(data)

        child = SubsetProblem(mlpb,{},False, subset)
        print len(child)
        assert len(child) == 3

        new_data = np.arange(10).reshape((5,2))
        fullSized = child.apply_on(new_data)
        assert len(fullSized) == 5




class TestSubsetFieldsProblem:

    def test_len(self):
        """SubsetFieldsProblem length"""
        data = np.arange(20).reshape((4,5))
        fields = [0,1,4]
        sfpb = SubsetFieldsProblem(data, {}, False, fields)

        assert len(sfpb) == 4

    def test_iter(self):
        """SubsetFieldsProblem iteration"""
        data = np.arange(20).reshape((4,5))
        fields = [0,1,4]
        sfpb = SubsetFieldsProblem(data, {}, False, fields)

        results = results = np.array((0,1,4,5,6,9,10,11,14,15,16,19)).reshape(4,3)
        i=0
        for line in sfpb:
            assert np.array_equal(line, results[i])
            i+=1

    def test_apply_on(self):
        """SubsetFieldsProblem apply_on uses same subset"""
        data = np.arange(20).reshape((4,5))
        fields = [0,1,4]
        sfpb = SubsetFieldsProblem(data, {}, False, fields)

        data = np.arange(21).reshape((3,7))
        mlpb = MLProblem(data)
        newProblem = sfpb.apply_on(mlpb)

        results = results = np.array((0,1,4,7,8,11,14,15,18)).reshape(3,3)
        i=0
        for line in newProblem:
            assert np.array_equal(line, results[i])
            i+=1




class TestMergedProblem:

    def test_len_serial(self):
        """MergedProblem length in serial mode"""
        data1 = np.arange(20).reshape((4,5))
        data2 = np.arange(30).reshape((3,10))
        data3 = np.arange(40).reshape((20,2))
        data4 = np.arange(50).reshape((5,10))

        mergedpb = MergedProblem([data1,data2,data3,data4],{}, False, True)

        assert len(mergedpb) == 32

    def test_len_not_serial(self):
        """MergedProblem length in normal mode"""
        data1 = np.arange(20).reshape((4,5))
        data2 = np.arange(30).reshape((3,10))
        data3 = np.arange(40).reshape((20,2))
        data4 = np.arange(50).reshape((5,10))

        mergedpb = MergedProblem([data1,data2,data3,data4],{}, False, False)
        assert len(mergedpb) == 80 #Max length * number of datasets

    def test_iter_serial(self):
        """MergedProblem iteration in serial mode"""
        data1 = np.arange(6).reshape((2,3))
        data2 = np.arange(9).reshape((3,3))
        data3 = np.arange(12).reshape((4,3))

        mergedpb = MergedProblem([data1,data2,data3],{}, False, True)

        results = np.array((0,1,2,3,4,5,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,9,10,11)).reshape(9,3)
        i = 0
        for line in mergedpb:
            assert np.array_equal(line,results[i])
            i+=1

    def test_iter_non_serial(self):
        """MergedProblem iteration in normal mode"""
        data1 = np.arange(6).reshape((2,3))
        data2 = np.arange(9).reshape((3,3))
        data3 = np.arange(12).reshape((4,3))

        mergedpb = MergedProblem([data1,data2,data3],{}, False, False)

        results = np.array((0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5,0,1,2,6,7,8,6,7,8,3,4,5,0,1,2,9,10,11)).reshape(12,3)
        i = 0
        for line in mergedpb:
            assert np.array_equal(line,results[i])
            i+=1




class TestPreprocessedProblem:

    def timesN(self, nombre, metadata):
        return metadata['scalar'] * nombre

    def test_iter(self):
        """PreprocessedProblem iteration"""
        data = np.arange(30).reshape((10,3))
        metadata = {'scalar': 4}
        pppb = PreprocessedProblem(data,metadata, True, self.timesN)

        results = 4 * data
        i=0
        for line in pppb:
            assert np.array_equal(line, results[i])
            i+=1

    def test_apply_on(self):
        """PreprocessedProblem apply_on applies the function on new data"""
        data = np.arange(30).reshape((10,3))
        metadata = {'scalar': 4}
        pppb = PreprocessedProblem(data,metadata, True, self.timesN)

        new_data = np.arange(40).reshape((5,8))
        new_metadata = {'scalar': 8}

        new_pppb = pppb.apply_on(new_data, new_metadata)

        results = 8 * new_data
        i=0
        for line in new_pppb:
            assert np.array_equal(line, results[i])
            i+=1
        
        # Second test, checker whether preprocess changes 
        # metadata properly
        data = np.arange(30).reshape((10,3))
        metadata = {'dummy': 3}
        def change_metadata(example,metadata):
            metadata['dummy'] = 10
        pppb = PreprocessedProblem(data,metadata, True, change_metadata)
        result = {'dummy':10}
        assert cmp(pppb.metadata,result) == 0

        new_data = np.arange(40).reshape((5,8))
        new_metadata = {'dummy': 6}
        new_pppb = pppb.apply_on(new_data, new_metadata)
        assert cmp(new_pppb.metadata,result) == 0



class TestMinibatchProblem:

    def test_metadata(self):
        """MinibatchProblem constructor sets 'minibatch_size' metadata"""
        data = np.arange(30).reshape((10,3))
        mppb = MinibatchProblem(data,{}, True, 2)

        assert 'minibatch_size' in mppb.metadata
        assert mppb.metadata['minibatch_size'] == 2

    def test_len(self):
        """MinibatchProblem length"""
        data = np.arange(30).reshape((10,3))
        mppb = MinibatchProblem(data,{}, True, 2)
        mppb2 = MinibatchProblem(data,{}, True, 3)

        assert len(mppb) == 5
        assert len(mppb2) == 4

    def test_iter_single_field(self):
        """MinibatchProblem iteration in single field mode"""
        data = np.arange(30).reshape((10,3))
        mppb = MinibatchProblem(data,{}, True, 2)
        mppb2 = MinibatchProblem(data,{}, True, 3)

        result = np.array((27,28,29,27,28,29,27,28,29)).reshape(3,3)
        i=0
        for line in mppb:
            assert len(line) == 2
        for line in mppb2:
            if i <3:
                assert len(line) == 3
            else:
                assert np.array_equal(line, result)
            i+=1

    def test_iter_multi_fields(self):
        """MinibatchProblem iteration in multi fields mode"""
        data = np.arange(30).reshape((10,3))
        mppb = MinibatchProblem(data,{}, True, 2, False)

        results = np.array((18, 21, 19, 22, 20, 23)).reshape(3,2)
        i = 0
        for ex in mppb:
            if i == 3:
                j=0
                for pair in ex:
                    assert np.array_equal(pair, results[j])
                    j+=1
            i+=1

    def test_apply_on_len(self):
        """MinibatchProblem apply_on doesn't change length"""
        data = np.arange(30).reshape((10,3))
        mppb = MinibatchProblem(data,{}, True, 2)

        new_data = np.arange(15).reshape((5,3))
        new_mppd = mppb.apply_on(new_data)

        assert len(new_mppd) == 3

    def test_apply_on_iter(self):
        """MinibatchProblem iteration after apply_on"""
        data = np.arange(30).reshape((10,3))
        mppb = MinibatchProblem(data,{}, True, 2)

        new_data = np.arange(15).reshape((5,3))
        new_mppd = mppb.apply_on(new_data)

        results = np.array((0,1,2,3,4,5)).reshape(2,3)
        assert np.array_equal(new_mppd.__iter__().next(), results)





class TestSemisupervisedProblem:

    def test_iter(self):
        """SemisupervisedProblem iteration"""
        data = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        unlabeled = set([0,2,4])
        sspb = SemisupervisedProblem(data,{},True,unlabeled)

        id = 0
        for line in sspb:
            if id in unlabeled:
                assert line[1] == None
            else:
                assert line[1] is not None
            id +=1

    def test_apply_on(self):
        """SemisupervisedProblem apply_on returns a basic MLProblem"""
        data = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        unlabeled = set([0,2,4])
        sspb = SemisupervisedProblem(data,{},True,unlabeled)

        new_data = [[0,1,2],[2,3,4],[4,5,6],[6,7,8],[8,9,10]]
        sspb2 = sspb.apply_on(new_data)

        for line in sspb2:
            assert None not in line
