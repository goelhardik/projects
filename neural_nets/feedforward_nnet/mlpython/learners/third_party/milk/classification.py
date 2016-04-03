# Copyright 2011 David Brouillard - Guillaume Roy-Fontaine. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY David Brouillard - Guillaume Roy-Fontaine ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL David Brouillard - Guillaume Roy-Fontaine OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of David Brouillard - Guillaume Roy-Fontaine.

"""
The ``learners.third_party.milk.classification`` module contains 
classifiers based on the Milk library:

* TreeClassifier:  Decision tree classifier.

"""


from mlpython.learners.generic import Learner
import numpy as np

try :
    import milk as libmilk
except ImportError:
    import warnings
    warnings.warn('\'import milk\' failed. The Milk library is not properly installed. See mlpython/learners/third_party/milk/README for instructions.')


class TreeClassifier(Learner):
    """ 
    Decision Tree Classifier using Milk library
 
    A decision tree classifier (currently, implements the greedy ID3
    algorithm without any pruning).

    Option ``criterion`` should be a string. Set it to
    ``'information_gain'``, to use the information gain splitting
    criterion (see
    http://en.wikipedia.org/wiki/Information_gain_in_decision_trees),
    or to ``'z1_loss'`` to use the 0-1 classification accuracy as the
    splitting criterion (default: ``'information_gain'``).

    Option ``min_split`` is a threshold, such that a node will not be
    split further if it has less than ``min_split`` examples in it
    (default: 4).

    If option ``include_entropy`` is True, the information gain criterion will
    include the original entropy (default: False).

    **Required metadata:**

    * ``'targets'``
    * ``'class_to_id'``

    **TODO:**

    * Support Milk's options ``R`` and ``subsample`` to support random sampling of splitting decisions.

    """
    def __init__(self, criterion='information_gain', min_split=4, include_entropy=False):
        self.criterion = criterion
        self.min_split = min_split
        self.include_entropy = include_entropy

        #self.subsample = subsample
        #self.R = R

    def train(self,trainset):
        """
        Trains the Milk Tree Learner.
        """
        
        self.n_classes = len(trainset.metadata['targets'])
        if self.n_classes > 2:
            raise ValueError('Invalid. Should have 2 classes.')
        
        features = np.zeros((len(trainset),trainset.metadata['input_size']))
        labels = np.zeros((len(trainset)),dtype='int')
        for i,xy in enumerate(trainset):
            x,y = xy
            features[i] = x
            labels[i] = y

        if self.criterion == 'information_gain':
            def criterion_fcn(labels0, labels1):
                return libmilk.supervised.tree.information_gain(labels0, labels1, include_entropy=self.include_entropy)
        elif self.criterion == 'z1_loss':
            def criterion_fcn(labels0, labels1):
                return libmilk.supervised.tree.z1_loss(labels0, labels1)
        else:
                raise ValueError('Invalid parameter: '+self.criterion+'. Should be either \'information_gain\' or \'z1_loss\'')

        learner = libmilk.supervised.tree_learner(criterion=criterion_fcn,min_split=self.min_split,return_label=True)
        #self.subsample = subsample
        #self.R = R
        model = learner.train(features, labels)
        
        self.tree = model

    def use(self,dataset):
        """
        Outputs the class predictions for ``dataset``.
        """
        features = []
        outputs = np.zeros((len(dataset),1))
        for xy in dataset:
            x,y = xy
            features += [x]

        for test,out in zip(features,outputs):
            out[0] = self.tree.apply(test)
        
        return outputs

    def forget(self):
        self.tree = None

    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
        the classification error cost for each example in the dataset.
        """
        outputs = self.use(dataset)
        
        costs = np.ones((len(outputs),1))
        # Compute classification error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            if y == pred[0]:
                cost[0] = 0

        return outputs,costs
        



