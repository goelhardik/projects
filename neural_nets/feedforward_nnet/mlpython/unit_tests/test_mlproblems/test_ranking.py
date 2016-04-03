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
The ``test_mlproblems.test_ranking`` module contains tests for MLProblems specifically
for ranking problems.

This module contains the following classes:

* TestRankingProblem:   .              Tests a ranking problem.
* TestRankingToClassificationProblem:  Tests a classification problem from a ranking problem.
* TestRankingToRegressionProblem:      Tests a regression problem from a ranking problem.

"""
import numpy as np
from mlpython.mlproblems.ranking import *
from nose.tools import *

class TestRankingProblem:
	"""Basic tests for RankingProblems"""

	data = ([[[  2.7,   1.0,   0.0], [  8.5,   7.5,   7.5]], [1, 0], 15],
		    [[[  1.7,   0.0,   0.0], [  6.5,   7.5,   7.5]], [1, 0], 10])

	def test_len_meta(self):
		"""Ranking problem 'n_queries' metadata should set the length"""
		metadata = {'n_queries':2}
		rpb = RankingProblem(self.data,metadata)

		assert len(rpb) == 2
		assert 'n_queries' not in rpb.metadata


	def test_iter(self):
		"""Ranking problem iteration"""
		rpb = RankingProblem(self.data)
		i=0
		for input, target, rank in rpb:
			if i == 1:
				print input
				assert np.array_equal(input,[([1.7,0.0,0.0],[  6.5,   7.5,   7.5])])
				assert np.array_equal(target, [[1,0]])
				assert rank == 10
			i+=1

	def test_len(self):
		"""Ranking problem length"""
		rpb = RankingProblem(self.data)
		assert len(rpb) == 2




class TestRankingToClassificationProblem:

	data = ([[[  2.7,   1.0,   0.0], [  8.5,   7.5,   7.5]], [1, 0], 15],
		    [[[  1.7,   0.0,   0.0], [  6.5,   7.5,   7.5]], [1, 0], 10])

	def test_len_meta(self):
		"""Ranking to classification length can be set with 'n_pairs'"""
		metadata = {'n_pairs':25,'scores':[3,2]}
		rcpb = RankingToClassificationProblem(self.data,metadata)

		assert len(rcpb) == 25
		assert 'n_pairs' not in rcpb.metadata

	@raises(KeyError)
	def test_no_scores(self):
		"""Ranking to classification needs 'scores' metadata. Expects KeyError"""
		rcpb = RankingToClassificationProblem(self.data)

	def test_setted_metadata(self):
		"""Ranking to classification should set 'class_to_id' and 'targets' metadatas"""
		metadata = {'scores':[3,2]}
		rcpb = RankingToClassificationProblem(self.data,metadata)

		assert 'class_to_id' in rcpb.metadata
		assert rcpb.metadata['class_to_id'] == {3:0,2:1}
		assert 'targets' in rcpb.metadata
		assert rcpb.metadata['targets'] == set([2,3])

	@raises(TypeError)
	def test_merge_missing_arguments(self):
		"""Ranking to classification merge function with one argument, expects TypeError"""
		def merge(a):
			return a**2

		metadata = {'scores':[3,2]}
		rcpb = RankingToClassificationProblem(self.data,metadata,True, merge)

		for a in rcpb:
			assert False

	def test_iter(self):
		"""Ranking to classification iteration"""
		def merge(a,b):
			return [a,b]
		metadata = {'scores':[3,0]}
		rcpb = RankingToClassificationProblem(self.data,metadata,True, merge)

		results_a = [[[2.7, 1.0, 0.0], 15],[[8.5, 7.5, 7.5], 15],[[1.7, 0.0, 0.0], 10],[[6.5, 7.5, 7.5], 10]]
		results_b = [None,1,None,1]
		i=0

		for a,b  in rcpb:
			assert a == results_a[i]
			assert b == results_b[i]
			i+=1



class TestRankingToRegressionProblem:

	data = ([[[  2.7,   1.0,   0.0], [  8.5,   7.5,   7.5]], [1, 0], 15],
		    [[[  1.7,   0.0,   0.0], [  6.5,   7.5,   7.5]], [1, 0], 10])

	def test_len_meta(self):
		"""Ranking to regression 'n_pairs' should set length"""
		metadata = {'n_pairs':25}
		rrpb = RankingToRegressionProblem(self.data,metadata)

		assert len(rrpb) == 25
		assert 'n_pairs' not in rrpb.metadata

	@raises(TypeError)
	def test_merge_missing_arguments(self):
		"""Ranking to regression merge function with one argument, expects TypeError"""
		def merge(a):
			return a**2

		rrpb = RankingToRegressionProblem(self.data,{},True, merge)

		for a in rrpb:
			assert False

	def test_iter(self):
		"""Ranking to regression iteration"""
		def merge(a,b):
			return [a,b]

		rrpb = RankingToRegressionProblem(self.data,{},True, merge)

		results_a = [[[2.7, 1.0, 0.0], 15],[[8.5, 7.5, 7.5], 15],[[1.7, 0.0, 0.0], 10],[[6.5, 7.5, 7.5], 10]]
		results_b = [1,0,1,0]
		i=0

		for a,b  in rrpb:
			assert a == results_a[i]
			assert b == results_b[i]
			i+=1
