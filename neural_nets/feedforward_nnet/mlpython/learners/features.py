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
The ``learners.features`` module contains Learners meant for feature
or representation learning. The MLProblems for these Learners should
be iterators over inputs. Their output should be a new feature
representation of the input.

The currently implemented algorithms are:

* PCA: Principal Component Analysis learner.
* RBM: Restricted Boltzmann Machine learner for feature extraction.

"""

from generic import Learner,OnlineLearner
import mlpython.mlproblems.generic as mlpb
import mlpython.mathutils.nonlinear as mlnonlin
import mlpython.mathutils.linalg as mllin
import numpy as np

class PCA(Learner):
    """
    Principal Component Analysis.

    Outputs the input's projection on the principal components, so as
    to obtain a representation with mean zero and identity covariance.

    Option ``n_components`` is the number of principal components to
    compute.

    Option ``regularizer`` is a small constant to add to the diagonal
    of the estimated covariance matrix (default=1e-10).

    **Required metadata:**

    * ``'input_size'``: Size of the inputs.

    """
    def __init__( self,
                  n_components,
                  regularizer=1e-10
                  ):
        self.n_components = n_components
        self.regularizer = regularizer

    def train(self,trainset):
        """
        Extract principal components.
        """

        # Put data in Numpy matrix
        input_size = trainset.metadata['input_size']
        trainmat = np.zeros((len(trainset),input_size))
        t = 0
        for input in trainset:
            trainmat[t,:] = input
            t+=1

        # Compute mean and covariance
        self.mean = trainmat.mean(axis=0)
        train_cov = np.cov(trainmat,rowvar=0)
        # Add a small constant on the diagonal, to regularize
        train_cov += np.diag(self.regularizer*np.ones(input_size))

        ## Compute principal components
        w,v = np.linalg.eigh(train_cov)
        s = (-w).argsort()
        w = w[s]
        v = v[:,s]

        self.transform = (1./np.sqrt(w[:self.n_components])).reshape((1,-1))*v[:,:self.n_components]

    def forget(self):
        del self.transform
        del self.mean

    def use(self,dataset):
        """
        Outputs the projection on the principal components, so as to obtain
        a representation with mean zero and identity covariance.
        """
        return [ np.dot(input-self.mean,self.transform) for input in dataset ]

    def test(self,dataset):
        """
        Outputs the squared error of the reconstructed inputs.
        """
        outputs = self.use(dataset)
        costs = zeros(len(dataset),1)
        for input,output,cost in zip(dataset,outputs,costs):
            cost[0] = np.sum((input-self.mean -np.dot(output,self.transform.T))**2)

        return outputs,costs

class RBM(OnlineLearner):
   """
   Restricted Boltzmann Machine for feature learning

   Option ``n_stages`` is the number of training iterations.

   Options ``learning_rate`` and ``decrease_constant`` correspond
   to the learning rate and decrease constant used for stochastic
   gradient descent.

   Option ``hidden_size`` should be a positive integer specifying
   the number of hidden units (features).
   
   Option ``l1_regularization`` is the weight of L1 regularization on
   the connection matrix.

   Option ``seed`` determines the seed for randomly initializing the
   weights.

   **Required metadata:**
   
   * ``'input_size'``: Size of the inputs.

   """

   def __init__(self, n_stages, 
                learning_rate = 0.01, 
                decrease_constant = 0,
                hidden_size = 100,
                l1_regularization = 0,
                seed = 1234
                ):
       self.n_stages = n_stages
       self.stage = 0
       self.learning_rate = learning_rate
       self.decrease_constant = decrease_constant
       self.hidden_size = hidden_size
       self.l1_regularization = l1_regularization
       self.seed = seed

   def initialize_learner(self,metadata):
      self.rng = np.random.mtrand.RandomState(self.seed)
      self.input_size = metadata['input_size']
      if self.hidden_size <= 0:
          raise ValueError('hidden_size should be > 0')

      self.W = (2*self.rng.rand(self.hidden_size,self.input_size)-1)/self.input_size
      self.c = np.zeros((self.hidden_size))
      self.b = np.zeros((self.input_size))

      self.deltaW = np.zeros((self.hidden_size,self.input_size))
      self.deltac = np.zeros((self.hidden_size))
      self.deltab = np.zeros((self.input_size))

      self.input = np.zeros((self.input_size))
      self.hidden = np.zeros((self.hidden_size))
      self.hidden_act = np.zeros((self.hidden_size))
      self.hidden_prob = np.zeros((self.hidden_size))

      self.neg_input = np.zeros((self.input_size))
      self.neg_input_act = np.zeros((self.input_size))
      self.neg_input_prob = np.zeros((self.input_size))
      self.neg_hidden_act = np.zeros((self.hidden_size))
      self.neg_hidden_prob = np.zeros((self.hidden_size))

      self.neg_stats = np.zeros((self.hidden_size,self.input_size))

      self.n_updates = 0

   def update_learner(self,example):
      self.input[:] = example

      # Performing CD-1
      mllin.product_matrix_vector(self.W,self.input,self.hidden_act)
      self.hidden_act += self.c
      mlnonlin.sigmoid(self.hidden_act,self.hidden_prob)
      np.less(self.rng.rand(self.hidden_size),self.hidden_prob,self.hidden)

      mllin.product_matrix_vector(self.W.T,self.hidden,self.neg_input_act)
      self.neg_input_act += self.b
      mlnonlin.sigmoid(self.neg_input_act,self.neg_input_prob)
      np.less(self.rng.rand(self.input_size),self.neg_input_prob,self.neg_input)

      mllin.product_matrix_vector(self.W,self.neg_input,self.neg_hidden_act)
      self.neg_hidden_act += self.c
      mlnonlin.sigmoid(self.neg_hidden_act,self.neg_hidden_prob)

      mllin.outer(self.hidden_prob,self.input,self.deltaW)
      mllin.outer(self.neg_hidden_prob,self.neg_input,self.neg_stats)
      self.deltaW -= self.neg_stats

      np.subtract(self.input,self.neg_input,self.deltab)
      np.subtract(self.hidden_prob,self.neg_hidden_prob,self.deltac)

      self.deltaW *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
      self.deltab *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
      self.deltac *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)         

      self.W += self.deltaW
      self.b += self.deltab
      self.c += self.deltac

      if self.l1_regularization > 0:
         self.W *= (np.abs(self.W) > (self.l1_regularization * self.learning_rate/(1.+self.decrease_constant*self.n_updates)))

      self.n_updates += 1

   def use_learner(self,example):
      output = np.zeros((self.hidden_size))
      mllin.product_matrix_vector(self.W,example,self.hidden_act)
      self.hidden_act += self.c
      mlnonlin.sigmoid(self.hidden_act,output)

      return [output]

   def cost(self,outputs,example):
      hidden = outputs[0]
      mllin.product_matrix_vector(self.W.T,hidden,self.neg_input_act)
      self.neg_input_act += self.b
      mlnonlin.sigmoid(self.neg_input_act,self.neg_input_prob)
      
      return [ np.sum((example-self.neg_input_prob)**2) ]

