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
The ``mlproblems.ranking`` module contains MLProblems specifically
for ranking problems.

This module contains the following classes:

* RankingProblem:   .              Generates a ranking problem.
* RankingToClassificationProblem:  Generates a classification problem from a ranking problem.
* RankingToRegressionProblem:      Generates a regression problem from a ranking problem.

"""

import generic as mlpb
import numpy as np

class RankingProblem(mlpb.MLProblem):
    """
    Generates a ranking problem.

    The data should be an iterator input/target/query pairs, where the
    target is a relevance score. When grouping query data together,
    is it assumed that examples from the same query are next to each
    other in the data (e.g. there aren't examples from the same query
    at the beginning and end of the data).

    The ranking examples become triplets
    (inputs_for_query,targets_for_query,query) where inputs_for_query
    and target_for_query are the lists of inputs and targets for a
    given query.

    **Required metadata:**
    
    * ``'n_queries'``: Number of queries (optional, will set the output of ``__len__(self)``).

    """

    def __init__(self, data=None, metadata={},call_setup=True):
        mlpb.MLProblem.__init__(self,data,metadata)
        self.__length__ = None
        if 'n_queries' in self.metadata:  # Gives a chance to set length through metadata
            self.__length__ = self.metadata['n_queries']
            del self.metadata['n_queries'] # So that it isn't passed to subsequent mlproblems

        if call_setup: RankingProblem.setup(self)

    def __iter__(self):
        tot_input = []
        tot_target = []
        last_query = None

        for input,target,query in self.data:
            if last_query != None: # Is not first example
                if not np.all(last_query == query): # Yield ranking example if query changed
                    yield (tot_input,tot_target,last_query)
                    tot_input = []
                    tot_target = []
            tot_input += [input]
            tot_target += [target]
            last_query = query

        if tot_input: # Output last ranking example
            yield (tot_input,tot_target,last_query)

class RankingToClassificationProblem(mlpb.MLProblem):
    """
    Generates a classification problem from a ranking problem.

    Option ``'merge_document_and_query'`` (a function with 2 arguments)
    is used to generate inputs for the classification problem.
    
    The list of possible scores must be provided as a metadata.
    **IMPORTANT:** the scores should be ordered from less to more relevant
    in the list. This list will be used to generate the set of targets
    and 'class_to_id' mapping.

    **Required_metadata:**

    * ``'scores'``: List of possible scores, ordered from less relevant to more relevant.
    * ``'n_pairs'``: Number of document/query pairs (optional, will set the output of ``__len__(self)``).

    **Defined metadata:**
    
    * ``'targets'``
    * ``'class_to_id'``

    """

    def __init__(self, data=None, metadata={},call_setup=True,merge_document_and_query=None):
        mlpb.MLProblem.__init__(self,data,metadata)
        self.merge_document_and_query = merge_document_and_query

        self.__length__ = None
        if 'n_pairs' in self.metadata:  # Gives a chance to set length through metadata
            self.__length__ = self.metadata['n_pairs']
            del self.metadata['n_pairs'] # So that it isn't passed to subsequent mlproblems

        if call_setup: RankingToClassificationProblem.setup(self)

    def __iter__(self):
        for inputs,targets,query in self.data:
            for input,target in zip(inputs,targets):
                if target in self.class_to_id:
                    yield self.merge_document_and_query(input,query),self.class_to_id[target]
                else:
                    yield self.merge_document_and_query(input,query),None # For unlabeled data

    def setup(self):
        # Creating class (string) to id (integer) mapping
        self.class_to_id = {}
        current_id = 0
        for score in self.metadata['scores']:
            self.class_to_id[score] = current_id
            current_id += 1
        self.metadata['class_to_id'] = self.class_to_id
        self.metadata['targets'] = set(self.metadata['scores'])

    def apply_on(self, new_data, new_metadata=None):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = RankingToClassificationProblem(new_data,new_metadata,call_setup=False,
                                                     merge_document_and_query=self.merge_document_and_query)
        new_problem.metadata['class_to_id'] = self.metadata['class_to_id']
        new_problem.metadata['targets'] = self.metadata['targets']
        new_problem.class_to_id = self.class_to_id
        return new_problem


class RankingToRegressionProblem(mlpb.MLProblem):
    """
    Generates a regression problem from a ranking problem.

    Option ``'merge_document_and_query'`` (a function with 2 arguments)
    is used to generate inputs for the classification problem.
    
    **Required_metadata:**

    * ``'n_pairs'``: Number of document/query pairs (optional, will set the output of ``__len__(self)``).

    """

    def __init__(self, data=None, metadata={},call_setup=True,merge_document_and_query=None):
        mlpb.MLProblem.__init__(self,data,metadata)
        self.merge_document_and_query = merge_document_and_query

        self.__length__ = None
        if 'n_pairs' in self.metadata:  # Gives a chance to set length through metadata
            self.__length__ = self.metadata['n_pairs']
            del self.metadata['n_pairs'] # So that it isn't passed to subsequent mlproblems

        if call_setup: RankingToRegressionProblem.setup(self)

    def __iter__(self):
        for inputs,targets,query in self.data:
            for input,target in zip(inputs,targets):
                yield self.merge_document_and_query(input,query),target

    def apply_on(self, new_data, new_metadata=None):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = RankingToRegressionProblem(new_data,new_metadata,call_setup=False,
                                                     merge_document_and_query=self.merge_document_and_query)
        return new_problem
