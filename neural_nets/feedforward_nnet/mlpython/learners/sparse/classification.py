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
Learners in this module are meant for classification problems on sparse data. 
They normally will require (at least) the metadata ``'targets'``.
The mlproblems for these learners should be iterators over pairs
of inputs and targets, with the target being a class index.

The currently implemented algorithms are:

* MultinomialNaiveBayesClassifier: a multinomial naive Bayes classifier.

"""

from mlpython.learners.generic import Learner
import numpy as np

class MultinomialNaiveBayesClassifier(Learner):
    """
    Multinomial Naive Bayes Classifier.

    This simple classifier has been found useful for text classification.
    Each non-zero input feature is treated as indication of the presence
    of a word, and its value is treated as the frequency of that word.

    Options ``dirichlet_prior_parameter`` controls the amount of regularization.

    **Required metadata:**

    * ``'targets'``
    * ``'input_size'``

    | **Reference:** 
    | A Comparison of Event Models for Naive Bayes Text Classification
    | McCallum and Nigam
    | http://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf

    """

    def __init__(self, dirichlet_prior_parameter=1):
        self.dirichlet_prior_parameter = dirichlet_prior_parameter

    def train(self, trainset):
        self.input_size = trainset.metadata['input_size']
        self.n_classes = len(trainset.metadata['targets'])

        # Initialize the model
        self.p_w_given_c = np.ones((self.input_size,self.n_classes))*self.dirichlet_prior_parameter
        self.p_c = np.zeros((self.n_classes))

        # Train the model
        for input,target in trainset:
            values,indices = input
            self.p_w_given_c[indices,target] += values
            self.p_c[target] += 1

        # Normalize counts
        self.p_w_given_c /= np.sum(self.p_w_given_c,0)
        self.log_p_w_given_c = np.log(self.p_w_given_c)
        self.p_c /= np.sum(self.p_c)
        self.log_p_c = np.log(self.p_c)

    def forget(self):        
        self.p_w_given_c[:] = 1./self.dirichlet_prior_parameter
        self.p_c[:] = 1./self.n_classes

    def use(self,dataset):
        probs = np.zeros((len(dataset),self.n_classes))
        count = 0
        outputs = []
        for example in dataset:
            values,indices = example[0]
            probs[count,:] = np.dot(values,self.log_p_w_given_c[indices,:])+self.log_p_c
            max_output = np.max(probs[count,:])
            probs[count,:] -= max_output
            probs[count,:] = np.exp(probs[count,:])
            probs[count,:] /= np.sum(probs[count,:])
            pred = np.argmax(probs[count,:])
            outputs += [[pred,probs[count,:]]]
            count += 1
        return outputs

    def test(self,dataset):
        outputs = self.use(dataset)
        costs = np.zeros((len(outputs),1))
        count = 0
        for input,target in dataset:
            pred = int(np.argmax(outputs[count,:]))
            costs[count,:] = int(pred!=target)
            count += 1

        return outputs,costs
