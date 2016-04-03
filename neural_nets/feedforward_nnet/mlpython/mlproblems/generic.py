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
The ``mlproblems.generic`` module contains MLProblems that are not
designed for a specific type of problem. They typically allow for
manipulations that can be useful for many tasks.

This module contains the following classes:

* MLProblem:              Root class for machine learning problems.
* SubsetProblem:          Extracts a subset of examples from a dataset.
* SubsetFieldsProblem:    Extracts a subset of the fields in a dataset.
* MergedProblem:          Merges several datasets together.
* PreprocessedProblem:    Applies an arbitrary preprocessing on a dataset.
* MinibatchProblem:       Puts examples of datasets into mini-batches.
* SemisupervisedProblem:  Removes the labels of a subset of the examples in a dataset.

"""

import numpy as np
import copy

class MLProblem:
    """
    Root class for machine learning problems.
    
    An MLProblem consists simply in an iterator over elements in 
    ``data``. It also has some metadata, or "data about the data".
    All that is assume about ``data`` is that it is possible to 
    iterate over its content.

    The metadata can be given explicitly by the user in the
    constructor. If ``data`` is itself an MLProblem, then its metadata
    will also be used (with priority given to the explicitly passed
    metadata).

    **Required metadata:**

    * ``'length'``: Number of examples (optional, will set the output of ``__len__(self)``).

    """

    def __init__(self, data=None, metadata={},call_setup=True):
        self.data = data
        self.metadata = {}
        if isinstance(data,MLProblem): # Use metadata from data if is an mlproblem
            self.metadata.update(data.metadata)
            self.__source_mlproblem__ = data
        else:
            self.__source_mlproblem__ = None
        self.metadata.update(metadata)

        self.__length__ = None
        if 'length' in self.metadata:  # Gives a chance to set length through metadata
            self.__length__ = self.metadata['length']
            del self.metadata['length'] # So that it isn't passed to subsequent mlproblems

        if call_setup: MLProblem.setup(self)

    def __iter__(self):
        for example in self.data:
            yield example

    def __len__(self):
        if self.__length__ is None: # if metadata hasn't been used to set length, use len(data)
            try:
                return len(self.data)
            except AttributeError:
                # Figure out length with exhaustive counting
                print 'Warning in mlpython.mlproblems.generic.MLProblem: couldn\'t get length from len(data)... will loop over MLProblem to compute length'
                self.__length__ = 0
                for example in self:
                    self.__length__ += 1
                return self.__length__
        else:
            return self.__length__

    def setup(self):
        """
        Adapts the MLProblem to the given data's content. For this
        root class, it does nothing.

        However, an MLProblem that would normalize examples by subtracting the
        data's average would compute this average in this method.
        """
        pass

    def apply_on(self, new_data, new_metadata={}):
        """
        Returns a new MLProblem that will apply on some new data the
        same processing that this MLProblem applies on its
        ``data``. For this root class, there isn't any processing to
        share, hence this method doesn't do much, besides calling
        ``data.apply_on(new_data,new_metadata)`` if ``data`` is itself
        an MLProblem.

        However, for an MLProblem that would normalize examples by subtracting the
        data's average, it would construct a new MLProblem such that it'll subtract the
        same average.
        """
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = self.__class__(new_data,new_metadata,call_setup=False)
        return new_problem
        
    def peak(self):
        """
        Returns the first example of the MLProblem.
        """
        return self.__iter__().next()

    def raw_source(self):
        """
        Returns the data and metadata of the first MLProblem in the
        series that led to this MLProblem.
        """
        if self.__source_mlproblem__ is None:
            return self.data,self.metadata
        else:
            return self.__source_mlproblem__.raw_source()

class SubsetProblem(MLProblem):
    """
    Extracts a subset of the examples in a dataset.
    
    The examples that are extracted have their ID (i.e. the example number
    from 0 to ``len(data)-1``, as defined by the order
    in which the iterator yields the examples) in a given ``subset``.

    """

    def __init__(self, data=None, metadata={},call_setup=True,subset=set([])):
        MLProblem.__init__(self,data,metadata)
        self.subset = subset
        if call_setup: SubsetProblem.setup(self)

    def __iter__(self):
        id = 0
        for example in self.data:
            if id in self.subset:
               yield example
            id += 1

    def __len__(self):
        return len(self.subset)

    def apply_on(self, new_data, new_metadata={}):
        # Since new_data probably doesn't use the same subset of example ids,
        # we either return a basic mlproblem or the output from the source mlproblem
        if self.__source_mlproblem__ is not None:
            new_problem = self.__source_mlproblem__.apply_on(new_data,new_metadata)
        else:
            new_problem = MLProblem(new_data,new_metadata,call_setup=False)

        return new_problem
        
class SubsetFieldsProblem(MLProblem):
    """
    Extracts a subset of the fields in a dataset.
    
    The fields that are selected are given by option
    ``fields``, a list of indices corresponding to
    the fields to keep. Each example of the new dataset
    will now be a list of those fields, unless ``fields``
    contains only one index, in which case each example will
    correspond to that field.

    """

    def __init__(self, data=None,metadata={},call_setup=True,fields=[0]):
        MLProblem.__init__(self,data,metadata)
        self.fields = fields
        if call_setup: SubsetFieldsProblem.setup(self)

    def __iter__(self):
        for example in self.data:
            if len(self.fields) == 1:
                yield example[self.fields[0]]
            else:
                yield [example[i] for i in self.fields]

    def apply_on(self, new_data, new_metadata={}):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = SubsetFieldsProblem(new_data,new_metadata,call_setup=False,fields=self.fields)
        return new_problem

class MergedProblem(MLProblem):
    """
    Merges several datasets together.
    
    Each element of data should itself be an iterator
    over examples. All examples of the first dataset
    are first iterated over, then all examples of the second,
    and so on.

    If option ``serial`` is False, then instead of iterating 
    over the examples of one dataset at a time, it cycles 
    over datasets and each time returns only one example.
    The iterator stops when all examples in all datasets
    have been iterated over at least once. Notice that
    if the datasets don't all have the same size, then
    some examples will be iterated over at least twice.

    """

    def __init__(self, data=None, metadata={},call_setup=True,serial=True):
        self.data = data
        self.metadata = {}
        if isinstance(data[0],MLProblem): # Use metadata from data if is an mlproblem
            self.metadata.update(data[0].metadata)
            self.__source_mlproblem__ = data[0]
        else:
            self.__source_mlproblem__ = None
        self.metadata.update(metadata)

        #self.__length__ = None
        #if 'length' in self.metadata:  # Gives a chance to set length through metadata
        #    self.__length__ = self.metadata['length']
        #    del self.metadata['length'] # So that it isn't passed to subsequent mlproblems

        self.serial = serial
        if call_setup: MergedProblem.setup(self)

    def __iter__(self):
        if self.serial:
            for dataset in self.data:
                for example in dataset:
                    yield example
        else:
            iterated_over_once = [False]*len(self.data)
            # Initialize iterators
            iterators = [dataset.__iter__() for dataset in self.data]
            examples = [ iter.next() for iter in iterators ]
            while not all(iterated_over_once):
                for example in examples:
                    yield example
                for t,iter in enumerate(iterators):
                    try:
                        example = iter.next() 
                    except StopIteration:
                        iterators[t] = self.data[t].__iter__()
                        iterated_over_once[t] = True
                        example = iterators[t].next()
                    examples[t] = example

    def __len__(self):
        if self.serial:
            l = 0
            for dataset in self.data:
                l += len(dataset)
            return l
        else:
            max_l = max([len(dataset) for dataset in self.data])
            return max_l * len(self.data)
    
    def apply_on(self, new_data, new_metadata={}):
        # Since new_data is probably not a list of mlproblems, 
        # we either return a basic mlproblem or the output from the source mlproblem
        if self.__source_mlproblem__ is not None:
            new_problem = self.__source_mlproblem__.apply_on(new_data,new_metadata)
        else:
            new_problem = MLProblem(new_data,new_metadata,call_setup=False)

        return new_problem

class PreprocessedProblem(MLProblem):
    """
    MLProblem that applies a preprocessing function on examples from a dataset.

    The examples of this MLProblem is the result
    of applying option ``preprocess`` on the examples
    in the original data. Hence, ``preprocess`` should
    be a callable function taking two arguments (an 
    example from the original data as well as the metadata) 
    and returning a preprocessed example.

    **IMPORANT:** if ``preprocess`` changes the size of the inputs, 
    the metadata (i.e. ``'input_size'``) should be changed 
    accordingly within ``preprocess``.

    """

    def __init__(self, data=None, metadata={},call_setup=True,preprocess=None):
        MLProblem.__init__(self,data,metadata)
        self.preprocess = preprocess
        if call_setup: PreprocessedProblem.setup(self)

        # Call preprocess on first example, so that it sets the new_metadata correctly
        self.__iter__().next()

    def __iter__(self):
        for example in self.data:
            yield self.preprocess(example,self.metadata)

    def apply_on(self, new_data, new_metadata={}):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = PreprocessedProblem(new_data,new_metadata,call_setup=False,preprocess=self.preprocess)
        
        # Call preprocess on first example, so that it sets the new_metadata correctly
        new_problem.__iter__().next()
        
        return new_problem

class MinibatchProblem(MLProblem):
    """
    MLProblem that puts examples into mini-batches.

    Option ``minibatch_size`` determines the size of the mini-batches.
    By default, this class assumes that the underlying dataset corresponds
    to a single field (e.g. the input). If this is not the case (e.g.
    contains pairs of inputs and targets), option ``has_single_field``
    should be set to ``False``.

    If the examples don't fit evenly into mini-batches of the desired
    size, the last mini-batch will be filled with copies of the
    remaining examples.

    **Defined metadata:**

    * ``'minibatch_size'``: number of examples in each mini-batch

    """

    def __init__(self, data=None, metadata={},call_setup=True,minibatch_size=None,
                 has_single_field=True):
        MLProblem.__init__(self,data,metadata)
        self.minibatch_size = minibatch_size
        self.has_single_field = has_single_field
        if call_setup: MinibatchProblem.setup(self)
        self.metadata['minibatch_size'] = self.minibatch_size

    def __len__(self):
        return int(np.ceil(float(len(self.data))/self.minibatch_size))

    def __iter__(self):
        minibatch_filling_count = 0
        for example in self.data:
            if minibatch_filling_count == 0:
                if self.has_single_field:
                    if (not hasattr(example,'shape')) or example.shape == (1,):
                        minibatch_container = np.zeros((self.minibatch_size,))
                    else:
                        minibatch_container = np.zeros((self.minibatch_size,)+example.shape)
                else:
                    minibatch_container = ()
                    for field in example:
                        if (not hasattr(field,'shape')) or field.shape == (1,):
                            minibatch_container += (np.zeros((self.minibatch_size,)),)
                        else:
                            minibatch_container += (np.zeros((self.minibatch_size,)+field.shape),)
            
            if self.has_single_field:
                minibatch_container[minibatch_filling_count] = example
            else:
                for f in range(len(minibatch_container)):
                    minibatch_container[f][minibatch_filling_count] = example[f]

            minibatch_filling_count += 1
            if minibatch_filling_count == self.minibatch_size:
                yield minibatch_container
                minibatch_filling_count = 0

        if minibatch_filling_count > 0:
            if self.has_single_field:
                i = 0
                while minibatch_filling_count < self.minibatch_size:
                    minibatch_container[minibatch_filling_count] = minibatch_container[i]
                    i+=1
                    minibatch_filling_count+=1
            else:
                i = 0
                while minibatch_filling_count < self.minibatch_size:
                    for f in range(len(minibatch_container)):
                        minibatch_container[f][minibatch_filling_count] = minibatch_container[f][i]
                    i+=1
                    minibatch_filling_count+=1
            yield minibatch_container

    def apply_on(self, new_data, new_metadata={}):
        if self.__source_mlproblem__ is not None:
            new_data = self.__source_mlproblem__.apply_on(new_data,new_metadata)
            new_metadata = {}   # new_data should already contain the new_metadata, since it is an mlproblem

        new_problem = MinibatchProblem(new_data,new_metadata,call_setup=False,minibatch_size=self.minibatch_size,has_single_field=self.has_single_field)
        return new_problem

class SemisupervisedProblem(MLProblem):
    """
    Removes the labels of a subset of the examples in a dataset.
    
    The examples that have their ID (i.e. the example number
    from 0 to ``len(data)-1``, as defined by the order
    in which the iterator yields the examples) in ``unlabeled_ids``
    will have their labels be replaced by None.

    The index of the label field can be given by option ``label_field``.

    """

    def __init__(self, data=None, metadata={},call_setup=True,unlabeled_ids=set([]),label_field=1):
        MLProblem.__init__(self,data,metadata)
        self.unlabeled_ids = unlabeled_ids
        self.label_field = label_field
        if call_setup: SemisupervisedProblem.setup(self)

    def __iter__(self):
        id = 0
        for example in self.data:
            if id in self.unlabeled_ids:
                unlabeled_example = copy.deepcopy(example)
                unlabeled_example[self.label_field] = None
                yield unlabeled_example
            else:
                yield example
            id += 1

    def apply_on(self, new_data, new_metadata={}):
        # Don't apply the same unlabeling to new_data.
        # We either return a basic mlproblem or the output from the source mlproblem
        if self.__source_mlproblem__ is not None:
            new_problem = self.__source_mlproblem__.apply_on(new_data,new_metadata)
        else:
            new_problem = MLProblem(new_data,new_metadata,call_setup=False)

        return new_problem
