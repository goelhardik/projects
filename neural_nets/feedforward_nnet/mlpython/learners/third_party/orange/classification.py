# Copyright 2011 Hugo larochelle. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY Hugo larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo larochelle.

"""
The ``learners.third_party.orange.classification`` module contains 
classifiers from the Orange library:

* RandomForest:  Random forest classifier.
* BoostedTrees:  Ensemble of boosted trees (Adaboost.M1).

It also contains one helper function:

* make_orange_dataset:    converts an MLProblem into a classification dataset in Orange format.

"""


from mlpython.learners.generic import Learner
import numpy as np

try :
    import orange
except ImportError:
    import warnings
    warnings.warn('\'import orange\' failed. The Orange library is not properly installed. See mlpython/learners/third_party/orange/README for instructions.')

try :
    import orngEnsemble
except ImportError:
    import warnings
    warnings.warn('\'import orngEnsemble\' failed. The Orange library is not properly installed. See mlpython/learners/third_party/orange/README for instructions.')

try :
    import orngTree
except ImportError:
    import warnings
    warnings.warn('\'import orngTree\' failed. The Orange library is not properly installed. See mlpython/learners/third_party/orange/README for instructions.')

def make_orange_dataset(dataset,domain = None):
    """
    Returns a classification dataset into the Orange format. The
    domain of the dataset can be specified (default is None, in which
    case the domain is computed from the metadata).
    """

    if domain is None:
        classes = [ str(i) for i in range(len(dataset.metadata['targets'])) ]
        columns = tuple([ 'input_'+str(i) for i in range(dataset.metadata['input_size']) ])
        input_attr = map(orange.FloatVariable,columns)
        class_attr = orange.EnumVariable('y',values = classes)
        domain = orange.Domain(input_attr,class_attr)
    
    input_size = dataset.metadata['input_size']
    examples = np.zeros((len(dataset),input_size+1))
    for i,xy in enumerate(dataset):
        x,y = xy
        examples[i,:input_size] = x
        examples[i,input_size] = y

    return orange.ExampleTable(domain,examples)
    

class RandomForest(Learner):
    """ 
    Random Forest classifeir based on the Orange library.
 
    Option ``n_trees`` is the number of trees to train in the ensemble
    (default = 50).
    
    Option ``n_features_per_node`` is the number of inputs (features)
    to consider when splitting a tree node. The default (None) is
    to use the square root of the input size.

    Option ``seed`` will set the random number generator's seed.

    **Required metadata:**

    * ``'targets'``
    * ``'class_to_id'``

    """
    def __init__(self, n_trees = 50, 
                 n_features_per_node = None, 
                 seed = 1234):
        self.n_trees = n_trees
        self.n_features_per_node = n_features_per_node
        self.seed = seed


    def train(self,trainset):
        """
        Trains a random forest using TreeLearn.
        """
        
        self.n_classes = len(trainset.metadata['targets'])

        trainset_orange = make_orange_dataset(trainset)
        self.trainset_domain = trainset_orange.domain

        import random
        
        self.forest = orngEnsemble.RandomForestLearner(trees=self.n_trees, 
                                                       attributes = self.n_features_per_node,
                                                       rand = random.Random(self.seed),
                                                       name="forest")(trainset_orange)
        
    def use(self,dataset):
        """
        Outputs the class predictions for ``dataset``.
        """

        dataset_orange = make_orange_dataset(dataset,self.trainset_domain)
        outputs = np.zeros((len(dataset),1))
        for i in range(len(dataset)):
            outputs[i,0] = int(self.forest(dataset_orange[i]))
        return outputs

    def forget(self):
        self.forest = None

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
        
class BoostedTrees(Learner):
    """ 
    Ensemble of decision trees based on AdaBoost.M1.
 
    Option ``n_trees`` is the number of trees to train in the ensemble
    (default = 50).
    
    Option ``max_majority`` is the maximal proportion of the majority
    class. When this is exceeded, a node is not split further (default = 1.0).

    Option ``max_depth`` is the maximum depth of the trees (default = 2).

    Option ``min_leaf_size`` is a minimum threshold on the number of
    training examples in a node, below which a node is not split
    (default = 0).

    Option ``skip_prob`` is the probability of skipping an input when
    considering splits for a node (default = 0).

    **Required metadata:**

    * ``'targets'``
    * ``'class_to_id'``

    """
    def __init__(self, n_trees = 50, 
                 max_majority = 1.0,
                 max_depth = 2,
                 min_leaf_size = 0,
                 skip_prob = 0):
        self.n_trees = n_trees
        self.max_majority = max_majority
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.skip_prob = skip_prob

    def train(self,trainset):
        """
        Trains an ensemble of tree with Adaboost.M1.
        """
        
        self.n_classes = len(trainset.metadata['targets'])

        trainset_orange = make_orange_dataset(trainset)
        self.trainset_domain = trainset_orange.domain

        tree = orngTree.TreeLearner(max_majority=self.max_majority,
                                    max_depth=self.max_depth,
                                    min_instances=self.min_leaf_size,
                                    skip_prob=self.skip_prob)
        
        adaboost = orngEnsemble.BoostedLearner(learner=tree,
                                               t=self.n_trees, 
                                               name="AdaBoost.M1")
        self.boosted_trees = adaboost(instances=trainset_orange)
        
    def use(self,dataset):
        """
        Outputs the class predictions for ``dataset``.
        """

        dataset_orange = make_orange_dataset(dataset,self.trainset_domain)
        outputs = np.zeros((len(dataset),1))
        for i in range(len(dataset)):
            outputs[i,0] = int(self.boosted_trees(dataset_orange[i]))
        return outputs

    def forget(self):
        self.boosted_trees = None

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
        



