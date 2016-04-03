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
The ``learners.distribution`` module contains Learners meant for density or
distribution estimation problems.  The MLProblems for these Learners
should be iterators over inputs.

The currently implemented algorithms are:

* Bagdistribution: a distribution estimation learner where each example is a bag of inputs.
* NADE:       the Neural Autoregressive Distribution Estimator (NADE) for multivariate binary distribution estimation
* FVSBN:      a fully visible Sigmoid Belief Network (FVSBN) for binary distribution estimation

"""

from generic import Learner, OnlineLearner
import numpy as np
import mlpython.mlproblems.generic as mlpb
import mlpython.mathutils.nonlinear as mlnonlin
import mlpython.mathutils.linalg as mllin

class BagDistribution(Learner):
    """
    A distribution estimation learner where each example is a bag of inputs.

    Given a distribution learner (given by the user), this learner
    will train it on all inputs in all bags. It is
    assumed that the distribution learner outputs its estimate
    of the log-distribution (when calling ``use(...)``).

    """
    def __init__(   self,
                    estimator=None,# The distribution learner to be trained
                    ):
        self.stage = 0
        self.estimator = estimator

    def train(self,trainset):
        """
        Trains the estimator on all examples in all bags.
        Each call to train increments ``self.stage`` by 1.
        """

        self.distribution_trainset = mlpb.MergedProblem(data=trainset,metadata=trainset.metadata)
        self.estimator.train(self.distribution_trainset)
        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.estimator.forget()

    def use(self,dataset):
        """
        Outputs the sum of the distribution learning outputs for 
        all inputs in each bag (example).
        """
        outputs = np.zeros((len(dataset),1))
        for bag,pred in zip(dataset,outputs):
            out = 0
            for x in bag:
                out += self.estimator.use([x])[0]
            pred[0] = out
            
        return outputs

    def test(self,dataset):
        """
        Outputs the NLLs of each example, normalized
        by the size of the example's bag.
        """
        outputs = self.use(dataset)
        costs = zeros(len(dataset),1)
        for bag,o,c in zip(dataset,outputs,costs):
            c[0] = -o[0]/len(bag)

        return outputs,costs


class NADE(OnlineLearner):
   """
   Neural Autoregressive Distribution Estimator (NADE) for multivariate binary distribution estimation

   The options are:

   * ``n_stages``:           Number of training iterations.
   * ``learning_rate``:      Learning rate.
   * ``decrease_constant``:  Decrease constant.
   * ``untied_weights``:     Whether to untie the weights going into and out of the hidden units.
   * ``hidden_size``:        Number of hidden units.
   * ``input_order``:        List of integers corresponding to the order for input modeling.
   * ``seed``:               Seed for randomly initializing the weights.
   * ``alpha``:              Weight vector for each input generative cost.

   **Required metadata:**
   
   * ``'input_size'``

   | **Reference:** 
   | The Neural Autoregressive Distribution Estimator
   | Larochelle and Murray
   | http://www.cs.toronto.edu/~larocheh/publications/aistats2011_nade.pdf

   """

   def initialize_learner(self,metadata):
      self.rng = np.random.mtrand.RandomState(self.seed)
      self.input_size = metadata['input_size']
      if self.hidden_size <= 0:
          raise ValueError('hidden_size should be > 0')

      self.W = (2*self.rng.rand(self.hidden_size,self.input_size)-1)/self.input_size
      self.c = np.zeros((self.hidden_size))
      self.b = np.zeros((self.input_size))

      self.dW = np.zeros((self.hidden_size,self.input_size))
      self.dc = np.zeros((self.hidden_size))
      self.db = np.zeros((self.input_size))

      if self.untied_weights:
          self.V = (2*self.rng.rand(self.hidden_size,self.input_size)-1)/self.input_size
          self.dV = np.zeros((self.hidden_size,self.input_size))

      self.input = np.zeros((self.input_size))
      self.input_times_W = np.zeros((self.hidden_size,self.input_size))
      self.acc_input_times_W = np.zeros((self.hidden_size,self.input_size))
      self.hid = np.zeros((self.hidden_size,self.input_size))
      self.Whid = np.zeros((self.hidden_size,self.input_size))
      self.recact = np.zeros((self.input_size))
      self.rec = np.zeros((self.input_size))
      
      self.dinput_times_W = np.zeros((self.hidden_size,self.input_size))
      self.dacc_input_times_W = np.zeros((self.hidden_size,self.input_size))
      self.dhid = np.zeros((self.hidden_size,self.input_size))
      self.dWhid = np.zeros((self.hidden_size,self.input_size))
      self.dWenc = np.zeros((self.hidden_size,self.input_size))
      self.drecact = np.zeros((self.input_size))
      self.drec = np.zeros((self.input_size))
      
      self.n_updates = 0

   def update_learner(self,example):
      self.input[self.input_order] = example
   
      # fprop
      np.multiply(self.input,self.W,self.input_times_W)
      np.add.accumulate(self.input_times_W[:,:-1],axis=1,out=self.acc_input_times_W[:,1:])
      self.acc_input_times_W[:,0] = 0
      self.acc_input_times_W += self.c[:,np.newaxis]
      mlnonlin.sigmoid(self.acc_input_times_W,self.hid)

      if self.untied_weights:
          np.multiply(self.hid,self.V,self.Whid)
      else:
          np.multiply(self.hid,self.W,self.Whid)

      mllin.sum_columns(self.Whid,self.recact)
      self.recact += self.b
      mlnonlin.sigmoid(self.recact,self.rec)

      # bprop
      np.subtract(self.rec,self.input,self.drec)
      self.drec *= self.alpha
      self.db[:] = self.drec

      if self.untied_weights:
          np.multiply(self.drec,self.hid,self.dV)
          np.multiply(self.drec,self.V,self.dhid)
          self.dW[:] = 0
      else:
          np.multiply(self.drec,self.hid,self.dW)
          np.multiply(self.drec,self.W,self.dhid)

      mlnonlin.dsigmoid(self.hid,self.dhid,self.dacc_input_times_W)
      mllin.sum_rows(self.dacc_input_times_W,self.dc)      
      np.add.accumulate(self.dacc_input_times_W[:,:0:-1],axis=1,out=self.dWenc[:,-2::-1])
      self.dWenc[:,-1] = 0
      self.dWenc *= self.input
      self.dW += self.dWenc

      self.dW *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
      self.db *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
      self.dc *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)

      self.W -= self.dW
      self.b -= self.db
      self.c -= self.dc

      if self.untied_weights:
          self.dV *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
          self.V -= self.dV
      self.n_updates += 1

   def use_learner(self,example):
      self.input[self.input_order] = example
      output = np.zeros((self.input_size))
      recact = np.zeros((self.input_size))
   
      # fprop
      np.multiply(self.input,self.W,self.input_times_W)
      np.add.accumulate(self.input_times_W[:,:-1],axis=1,out=self.acc_input_times_W[:,1:])
      self.acc_input_times_W[:,0] = 0
      self.acc_input_times_W += self.c[:,np.newaxis]
      mlnonlin.sigmoid(self.acc_input_times_W,self.hid)
      if self.untied_weights:
          np.multiply(self.hid,self.V,self.Whid)
      else:
          np.multiply(self.hid,self.W,self.Whid)

      mllin.sum_columns(self.Whid,recact)
      recact += self.b
      mlnonlin.sigmoid(recact,output)
      return [output,recact]

   def cost(self,outputs,example):
      self.input[self.input_order] = example
      #return [ np.sum(-self.input*np.log(outputs[0]) - (1-self.input)*np.log(1-outputs[0])) ]
      return [ np.sum(-self.input*(outputs[1]-np.log(1+np.exp(outputs[1])))*self.alpha - (1-self.input)*(-outputs[1]-np.log(1+np.exp(-outputs[1])))*self.alpha) ]

   def sample(self):
      input = np.zeros(self.input_size)
      input_prob = np.zeros(self.input_size)
      hid_i = np.zeros(self.hidden_size)
      for i in range(self.input_size):
         if i > 0:
            mlnonlin.sigmoid(self.c+np.dot(self.W[:,:i],input[:i]),hid_i)
         else:
            mlnonlin.sigmoid(self.c,hid_i)

         if self.untied_weights:
            mlnonlin.sigmoid(np.dot(hid_i,self.V[:,i])+self.b[i:i+1],input_prob[i:i+1])
         else:
            mlnonlin.sigmoid(np.dot(hid_i,self.W[:,i])+self.b[i:i+1],input_prob[i:i+1])

         input[i] = (self.rng.rand()<input_prob[i])

      return (input[self.input_order],input_prob[self.input_order])

   def verify_gradients(self,untied_weights):
      
      print 'WARNING: calling verify_gradients reinitializes the learner'

      rng = np.random.mtrand.RandomState(1234)
      input_order = range(20)
      rng.shuffle(input_order)

      self.seed = 1234
      self.hidden_size = 10
      self.input_order = input_order
      self.untied_weights = untied_weights
      self.initialize_learner({'input_size':20})
      example = rng.rand(20)<0.5
      epsilon=1e-6
      self.learning_rate = 1
      self.decrease_constant = 0
      self.alpha = 1

      W_copy = np.array(self.W)
      emp_dW = np.zeros(self.W.shape)
      for i in range(self.W.shape[0]):
         for j in range(self.W.shape[1]):
            self.W[i,j] += epsilon
            output = self.use_learner(example)
            a = self.cost(output,example)[0]
            self.W[i,j] -= epsilon

            self.W[i,j] -= epsilon
            output = self.use_learner(example)
            b = self.cost(output,example)[0]
            self.W[i,j] += epsilon

            emp_dW[i,j] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.W[:] = W_copy
      print 'dW diff.:',np.sum(np.abs(self.dW.ravel()-emp_dW.ravel()))/self.W.ravel().shape[0]

      b_copy = np.array(self.b)
      emp_db = np.zeros(self.b.shape)
      for i in range(self.b.shape[0]):
         self.b[i] += epsilon
         output = self.use_learner(example)
         a = self.cost(output,example)[0]
         self.b[i] -= epsilon
         
         self.b[i] -= epsilon
         output = self.use_learner(example)
         b = self.cost(output,example)[0]
         self.b[i] += epsilon
         
         emp_db[i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.b[:] = b_copy
      print 'db diff.:',np.sum(np.abs(self.db.ravel()-emp_db.ravel()))/self.b.ravel().shape[0]

      c_copy = np.array(self.c)
      emp_dc = np.zeros(self.c.shape)
      for i in range(self.c.shape[0]):
         self.c[i] += epsilon
         output = self.use_learner(example)
         a = self.cost(output,example)[0]
         self.c[i] -= epsilon

         self.c[i] -= epsilon
         output = self.use_learner(example)
         b = self.cost(output,example)[0]
         self.c[i] += epsilon

         emp_dc[i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.c[:] = c_copy
      print 'dc diff.:',np.sum(np.abs(self.dc.ravel()-emp_dc.ravel()))/self.c.ravel().shape[0]

      if untied_weights:
         V_copy = np.array(self.V)
         emp_dV = np.zeros(self.V.shape)
         for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
               self.V[i,j] += epsilon
               output = self.use_learner(example)
               a = self.cost(output,example)[0]
               self.V[i,j] -= epsilon
         
               self.V[i,j] -= epsilon
               output = self.use_learner(example)
               b = self.cost(output,example)[0]
               self.V[i,j] += epsilon
         
               emp_dV[i,j] = (a-b)/(2.*epsilon)
         
         self.update_learner(example)
         self.V[:] = V_copy
         print 'dV diff.:',np.sum(np.abs(self.dV.ravel()-emp_dV.ravel()))/self.V.ravel().shape[0]


class FVSBN(OnlineLearner):
   """
   A fully visible Sigmoid Belief Network (FVSBN) for binary distribution estimation

   The options are:

   * ``n_stages``:           Number of training iterations.
   * ``learning_rate``:      Learning rate.
   * ``decrease_constant``:  Decrease constant.
   * ``input_order``:        List of integers corresponding to the order for input modeling.
   * ``seed``:               Seed for randomly initializing the weights.

   **Required metadata:**
   
   * ``'input_size'``

   | **Reference:** 
   | Connectionist Learning of Belief Networks
   | Neal

   """

   def initialize_learner(self,metadata):
      self.rng = np.random.mtrand.RandomState(self.seed)
      self.input_size = metadata['input_size']

      self.utri_index = []
      for i in range(self.input_size):
          for j in range(self.input_size):
              if i <= j:
                  self.utri_index += [i*self.input_size + j]

      self.W = (2*self.rng.rand(self.input_size,self.input_size)-1)/self.input_size
      self.W.ravel()[self.utri_index] = 0
      self.b = np.zeros((self.input_size))

      self.dW = np.zeros((self.input_size,self.input_size))
      self.db = np.zeros((self.input_size))

      self.input = np.zeros((self.input_size))
      self.input_times_W = np.zeros((self.input_size))
      self.recact = np.zeros((self.input_size))
      self.rec = np.zeros((self.input_size))
      
      self.drecact = np.zeros((self.input_size))
      self.drec = np.zeros((self.input_size))
      
      self.n_updates = 0

   def update_learner(self,example):
      self.input[self.input_order] = example
   
      # fprop
      mllin.product_matrix_vector(self.W,self.input,self.recact)
      self.recact += self.b
      mlnonlin.sigmoid(self.recact,self.rec)

      # bprop
      np.subtract(self.rec,self.input,self.drec)
      self.db[:] = self.drec
      mllin.outer(self.drec,self.input,self.dW)
      
      self.dW *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
      self.db *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)

      self.W -= self.dW
      self.b -= self.db

      self.W.ravel()[self.utri_index] = 0 # Setting back upper diagonal to 0
      self.n_updates += 1

   def use_learner(self,example):
      self.input[self.input_order] = example
      output = np.zeros((self.input_size))
      recact = np.zeros((self.input_size))
   
      # fprop
      mllin.product_matrix_vector(self.W,self.input,recact)
      recact += self.b
      mlnonlin.sigmoid(recact,output)
      return [output,recact]

   def cost(self,outputs,example):
      self.input[self.input_order] = example
      #return [ np.sum(-self.input*np.log(outputs[0]) - (1-self.input)*np.log(1-outputs[0])) ]
      return [ np.sum(-self.input*(outputs[1]-np.log(1+np.exp(outputs[1]))) - (1-self.input)*(-outputs[1]-np.log(1+np.exp(-outputs[1])))) ]

   def verify_gradients(self):
      
      print 'WARNING: calling verify_gradients reinitializes the learner'

      rng = np.random.mtrand.RandomState(1234)
      input_order = range(20)
      rng.shuffle(input_order)

      self.seed = 1234
      self.input_order = input_order
      self.initialize_learner({'input_size':20})
      example = rng.rand(20)<0.5
      epsilon=1e-6
      self.learning_rate = 1
      self.decrease_constant = 0

      W_copy = np.array(self.W)
      emp_dW = np.zeros(self.W.shape)
      for i in range(self.W.shape[0]):
         for j in range(self.W.shape[1]):
            self.W[i,j] += epsilon
            output = self.use_learner(example)
            a = self.cost(output,example)[0]
            self.W[i,j] -= epsilon

            self.W[i,j] -= epsilon
            output = self.use_learner(example)
            b = self.cost(output,example)[0]
            self.W[i,j] += epsilon

            emp_dW[i,j] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.W[:] = W_copy
      print 'dW diff.:',np.sum(np.abs(self.dW.ravel()-emp_dW.ravel()))/self.W.ravel().shape[0]


      b_copy = np.array(self.b)
      emp_db = np.zeros(self.b.shape)
      for i in range(self.b.shape[0]):
         self.b[i] += epsilon
         output = self.use_learner(example)
         a = self.cost(output,example)[0]
         self.b[i] -= epsilon
         
         self.b[i] -= epsilon
         output = self.use_learner(example)
         b = self.cost(output,example)[0]
         self.b[i] += epsilon
         
         emp_db[i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.b[:] = b_copy
      print 'db diff.:',np.sum(np.abs(self.db.ravel()-emp_db.ravel()))/self.b.ravel().shape[0]
