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
The ``learners.third_party.treelearn.regression`` module contains 
regression models from the TreeLearn library:

* RandomForest:  Random forest regression models.

"""


from mlpython.learners.generic import Learner
import numpy as np

try :
    import treelearn
except ImportError:
    import warnings
    warnings.warn('\'import treelearn\' failed. The TreeLearn library is not properly installed. See mlpython/learners/third_party/treelearn/README for instructions.')

class RandomForest(Learner):
    """ 
    Random Forest regression model based on the TreeLearn library
 
    Option ``n_trees`` is the number of trees to train in the ensemble
    (default = 50).

    Option ``additive`` is whether trees should be trained on the residual of the
    prediction from the previous trees (default = False).

    Option ``sample_percent`` is the proportion of the dataset to
    sample for training each tree (default = 0.5)

    Option ``n_features_per_node`` is the number of inputs (features)
    to consider when splitting a tree node. The default (None) is
    to use the log of the input size.

    Option ``min_leaf_size`` is a minimum threshold on the number of
    training examples in a node, below which a node is not split
    (default = 1).

    Option ``max_height`` is the maximum height of the trees 
    (default = 100).

    Option ``max_thresholds`` is the maximum number of thresholds to
    consider when splitting an input (feature). Those thresholds are
    evenly spaced between the minimum and maximum input value. The
    default (None) behavior is to consider all midpoints between unique input
    values.

    Option ``seed`` is the seed of the random number generator.

    **Required metadata:**

    * ``'input_size'``

    """
    def __init__(self, n_trees = 50, 
                 additive = False,
                 sample_percent = 0.5, 
                 n_features_per_node = None, 
                 min_leaf_size = 1, 
                 max_height = 100, 
                 max_thresholds = None,
                 seed = 1234):
        self.n_trees = n_trees
        self.additive = additive
        self.sample_percent = sample_percent
        self.n_features_per_node = n_features_per_node
        self.min_leaf_size = min_leaf_size
        self.max_height = max_height
        self.max_thresholds = max_thresholds
        self.seed = seed

        import random
        random.seed(self.seed)
        np.random.seed(self.seed)

    def train(self,trainset):
        """
        Trains a random forest using TreeLearn.
        """

        features = np.zeros((len(trainset),trainset.metadata['input_size']))
        labels = np.zeros((len(trainset)),dtype='int')
        for i,xy in enumerate(trainset):
            x,y = xy
            features[i] = x
            labels[i] = y

        base_tree = treelearn.RandomizedTree( num_features_per_node = self.n_features_per_node,
                                              min_leaf_size = self.min_leaf_size,
                                              max_height = self.max_height,
                                              max_thresholds = self.max_thresholds,
                                              regression = True)
        learner = treelearn.RegressionEnsemble(num_models = self.n_trees,
                                               bagging_percent = self.sample_percent,
                                               base_model = base_tree,
                                               additive = self.additive)

        learner.fit(features,labels)
        
        self.forest = learner

    def use(self,dataset):
        """
        Outputs the target predictions for ``dataset``.
        """
        features = []
        outputs = np.zeros((len(dataset),1))
        for xy in dataset:
            x,y = xy
            features += [x]

        outputs[:,0] = self.forest.predict(features)
        return outputs

    def forget(self):
        self.forest = None
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)

    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
        the regression error cost for each example in the dataset.
        """
        outputs = self.use(dataset)
        
        costs = np.ones((len(outputs),1))
        # Compute mean squared error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            cost[0] = (y-pred)**2

        return outputs,costs
        



