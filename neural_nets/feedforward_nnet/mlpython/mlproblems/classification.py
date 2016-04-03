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
The ``mlproblems.classification`` module contains MLProblems specifically
for classification problems.

This module contains the following classes:

* ClassificationProblem:   Generates a classification problem.
* ClassSubsetProblem:      Extracts examples from a subset of all classes.

"""

from generic import MLProblem

class ClassificationProblem(MLProblem):
    """
    Generates a classification problem.

    The data should be an iterator over input/target pairs. 
    
    **Required metadata:**
    
    * ``'targets'``: The set of possible values for the target.

    **Defined metadata:**

    * ``'class_to_id'``: A dictionary mapping from elements in ``'targets'`` 
      to a class id.

    """

    def __init__(self, data=None, metadata={},call_setup=True):
        MLProblem.__init__(self,data,metadata)
        if call_setup: ClassificationProblem.setup(self)

    def __iter__(self):
        for input,target in self.data:
            yield input,self.class_to_id[target]

    def setup(self):
        # Creating class (string) to id (integer) mapping
        self.class_to_id = {}
        current_id = 0
        for target in self.metadata['targets']:
            self.class_to_id[target] = current_id
            current_id += 1
        self.metadata['class_to_id'] = self.class_to_id

    def apply_on(self, new_data, new_metadata=None):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = ClassificationProblem(new_data,new_metadata,call_setup=False)
        new_problem.metadata['class_to_id'] = self.metadata['class_to_id']
        new_problem.class_to_id = self.class_to_id
        return new_problem

class ClassSubsetProblem(MLProblem):
    """
    Extracts examples in a dataset belonging to some subset of classes.
    
    Option ``subset`` gives the set of class symbols that should 
    be included. The metadata ``'class_to_id'`` that maps symbols
    to IDs is required (it is assumed that the targets have
    already been processed by this mapping, see ClassificationProblem).
   
    Option ``include_class`` determines whether to put the class ID
    in the example or only yield the input.

    **Required metadata:**
    
    * ``'class_to_id'``

    **Defined metadata:**

    * ``'class_to_id'``
    * ``'targets'``

    """

    def __init__(self, data=None, metadata={},call_setup=True,
                 subset=[], # Subset of classes to include
                 include_class=True # Whether to include the class field
                 ):
        MLProblem.__init__(self,data,metadata)
        self.subset=subset
        self.include_class = include_class

        self.__length__ = None
        if 'class_subset_length' in self.metadata:  # Gives a chance to set length through metadata
            self.__length__ = self.metadata['class_subset_length']
            del self.metadata['class_subset_length'] # So that it isn't passed to subsequent mlproblems
        else:
            # Since len(data) won't give the right answer, figure out what the length is by an exhaustive count
            parent_ids = set([])
            parent_class_to_id = self.metadata['class_to_id']
            for c in self.subset:
                parent_ids.add(parent_class_to_id[c])

            self.__length__ = 0
            for input,target in self.data:
                if target in parent_ids:
                    self.__length__+=1

        if call_setup: ClassSubsetProblem.setup(self) 

    def __iter__(self):
        for input,target in self.data:
            if target in self.parent_id_to_id:
                if self.include_class:
                    yield input,self.parent_id_to_id[target]
                else:
                    yield input

    def setup(self):
        self.class_to_id = {}
        self.parent_id_to_id = {}
        self.targets = set([])
        parent_class_to_id = self.metadata['class_to_id']
        id = 0
        for c in self.subset:
            self.class_to_id[c] = id
            self.parent_id_to_id[parent_class_to_id[c]]=id
            self.targets.add(c)
            id+=1
        self.metadata['targets'] = self.targets
        self.metadata['class_to_id'] = self.class_to_id

    def apply_on(self, new_data, new_metadata=None):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = ClassSubsetProblem(new_data,new_metadata,call_setup=False,subset=self.subset,
                                         include_class=self.include_class)
        new_problem.targets = self.targets
        new_problem.class_to_id = self.class_to_id
        new_problem.parent_id_to_id = self.parent_id_to_id
        new_problem.metadata['targets'] = self.targets
        new_problem.metadata['class_to_id'] = self.class_to_id
        return new_problem
        
