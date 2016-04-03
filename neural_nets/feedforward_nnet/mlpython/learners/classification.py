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
The ``learners.classification`` module contains Learners meant for classification problems. 
They normally will require (at least) the metadata ``'targets'``.
The MLProblems for these Learners should be iterators over pairs
of inputs and targets, with the target being a class index.

The currently implemented algorithms are:

* BayesClassifier: Bayes classifier obtained from distribution estimators.
* NNet:            Neural Network for classification.

"""

from generic import Learner,OnlineLearner
import numpy as np
import mlpython.mlproblems.classification as mlpb
import mlpython.mathutils.nonlinear as mlnonlin
import mlpython.mathutils.linalg as mllin


class BayesClassifier(Learner):
    """ 
    Bayes classifier from distribution estimators
 
    Given one distribution learner per class (option ``estimators``), this
    learner will train each one on a separate class and classify
    examples using Bayes' rule.

    **Required metadata:**
    
    * ``'targets'``

    """
    def __init__(   self,
                    estimators=[],# The distribution learners to be trained
                    ):
        self.stage = 0
        self.estimators = estimators

    def train(self,trainset):
        """
        Trains each estimator. Each call to train increments ``self.stage`` by 1.
        If ``self.stage == 0``, first initialize the model.
        """

        self.n_classes = len(trainset.metadata['targets'])

        # Initialize model
        if self.stage == 0:
            # Split data according to classes
            self.class_trainset = []
            tot_len = len(trainset)
            self.prior = np.zeros((self.n_classes))
            for c in xrange(self.n_classes):
                trainset_c = mlpb.ClassSubsetProblem(data=trainset,metadata=trainset.metadata,
                                                     subset=set([c]),
                                                     include_class=False)
                trainset_c.setup()
                self.class_trainset += [ trainset_c ]
                self.prior[c] = float(len(trainset_c))/tot_len

        # Training each estimators
        for c in xrange(self.n_classes):
            self.estimators[c].train(self.class_trainset[c])
        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        # Initialize estimators
        for c in xrange(self.n_classes):
            self.estimators[c].forget()
        self.prior = 1./self.n_classes * np.ones((self.n_classes))

    def use(self,dataset):
        """
        Outputs the class_id chosen by the algorithm, for each
        example in the dataset.
        """
        outputs = -1*np.ones((len(dataset),1))
        for xy,pred in zip(dataset,outputs):
            x,y = xy
            max_prob = -np.inf
            max_prob_class = -1
            for c in xrange(self.n_classes):
                prob_c = self.estimators[c].use([x])[0] + np.log(self.prior[c])
                if max_prob < prob_c:
                    max_prob = prob_c
                    max_prob_class = c
                
            pred[0] = max_prob_class
            
        return outputs

    def test(self,dataset):
        """
        Outputs the class_id chosen by the algorithm and
        the classification error cost for each example in the dataset
        """
        outputs = self.use(dataset)
        costs = np.ones((len(outputs),1))
        # Compute classification error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            if y == pred[0]:
                cost[0] = 0

        return outputs,costs


class NNet(OnlineLearner):
   """
   Neural Network for classification

   Option ``n_stages`` is the number of training iterations.

   Options ``learning_rate`` and ``decrease_constant`` correspond
   to the learning rate and decrease constant used for stochastic
   gradient descent.

   Option ``hidden_sizes`` should be a list of positive integers
   specifying the number of hidden units in each hidden layer, from
   the first to the last.
   
   Option ``seed`` determines the seed for randomly initializing the
   weights.

   Option ``pretrained_parameters`` should be a pair made of the
   list of hidden layer weights and biases, to replace random
   initialization. If None (default), random initialization will
   be used.

   **Required metadata:**

   * ``'input_size'``: Size of the input.
   * ``'targets'``: Set of possible targets.

   """

   def __init__(self, n_stages, 
                learning_rate = 0.01, 
                decrease_constant = 0,
                hidden_sizes = [ 100 ],
                seed = 1234,
                pretrained_parameters = None
                ):
       self.n_stages = n_stages
       self.stage = 0
       self.learning_rate = learning_rate
       self.decrease_constant = decrease_constant
       self.hidden_sizes = hidden_sizes
       self.seed = seed
       self.pretrained_parameters = pretrained_parameters

   def initialize_learner(self,metadata):
      self.n_classes = len(metadata['targets'])
      self.rng = np.random.mtrand.RandomState(self.seed)
      self.input_size = metadata['input_size']
      self.n_hidden_layers = len(self.hidden_sizes)
      if sum([nhid > 0 for nhid in self.hidden_sizes]) != self.n_hidden_layers:
         raise ValueError('All hidden layer sizes should be > 0')
      if self.n_hidden_layers < 1:
         raise ValueError('There should be at least one hidden layer')

      self.Ws = [(2*self.rng.rand(self.hidden_sizes[0],self.input_size)-1)/self.input_size]
      self.cs = [np.zeros((self.hidden_sizes[0]))]
      self.dWs = [np.zeros((self.hidden_sizes[0],self.input_size))]
      self.dcs = [np.zeros((self.hidden_sizes[0]))]

      self.layers = [np.zeros((self.input_size))]
      self.layer_acts = [np.zeros((self.input_size))]
      self.layers += [np.zeros((self.hidden_sizes[0]))]
      self.layer_acts += [np.zeros((self.hidden_sizes[0]))]

      self.dlayers = [np.zeros((self.input_size))]
      self.dlayer_acts = [np.zeros((self.input_size))]
      self.dlayers += [np.zeros((self.hidden_sizes[0]))]
      self.dlayer_acts += [np.zeros((self.hidden_sizes[0]))]

      for h in range(1,self.n_hidden_layers):
         self.Ws += [(2*self.rng.rand(self.hidden_sizes[h],self.hidden_sizes[h-1])-1)/self.hidden_sizes[h-1]]
         self.cs += [np.zeros((self.hidden_sizes[h]))]
         self.dWs += [np.zeros((self.hidden_sizes[h],self.hidden_sizes[h-1]))]
         self.dcs += [np.zeros((self.hidden_sizes[h]))]
         self.layers += [np.zeros((self.hidden_sizes[h]))]
         self.layer_acts += [np.zeros((self.hidden_sizes[h]))]
         self.dlayers += [np.zeros((self.hidden_sizes[h]))]
         self.dlayer_acts += [np.zeros((self.hidden_sizes[h]))]

      self.U = (2*self.rng.rand(self.n_classes,self.hidden_sizes[-1])-1)/self.hidden_sizes[-1]
      self.d = np.zeros((self.n_classes))
      self.dU = np.zeros((self.n_classes,self.hidden_sizes[-1]))
      self.dd = np.zeros((self.n_classes))
      self.output_act = np.zeros((self.n_classes))
      self.output = np.zeros((self.n_classes))
      self.doutput_act = np.zeros((self.n_classes))

      if self.pretrained_parameters is not None:
         self.Ws = self.pretrained_parameters[0]
         self.cs = self.pretrained_parameters[1]

      self.n_updates = 0

   def update_learner(self,example):
      self.layers[0][:] = example[0]

      # fprop
      for h in range(self.n_hidden_layers):
         mllin.product_matrix_vector(self.Ws[h],self.layers[h],self.layer_acts[h+1])
         self.layer_acts[h+1] += self.cs[h]
         mlnonlin.sigmoid(self.layer_acts[h+1],self.layers[h+1])

      mllin.product_matrix_vector(self.U,self.layers[-1],self.output_act)
      self.output_act += self.d
      mlnonlin.softmax(self.output_act,self.output)

      self.doutput_act[:] = self.output
      self.doutput_act[example[1]] -= 1
      self.doutput_act *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)

      self.dd[:] = self.doutput_act
      mllin.outer(self.doutput_act,self.layers[-1],self.dU)      
      mllin.product_matrix_vector(self.U.T,self.doutput_act,self.dlayers[-1])
      mlnonlin.dsigmoid(self.layers[-1],self.dlayers[-1],self.dlayer_acts[-1])
      for h in range(self.n_hidden_layers-1,-1,-1):
         self.dcs[h][:] = self.dlayer_acts[h+1]
         mllin.outer(self.dlayer_acts[h+1],self.layers[h],self.dWs[h])
         mllin.product_matrix_vector(self.Ws[h].T,self.dlayer_acts[h+1],self.dlayers[h])
         mlnonlin.dsigmoid(self.layers[h],self.dlayers[h],self.dlayer_acts[h])

      self.U -= self.dU
      self.d -= self.dd
      for h in range(self.n_hidden_layers-1,-1,-1):
         self.Ws[h] -= self.dWs[h]
         self.cs[h] -= self.dcs[h]

      self.n_updates += 1

   def use_learner(self,example):
      output = np.zeros((self.n_classes))
      self.layers[0][:] = example[0]

      # fprop
      for h in range(self.n_hidden_layers):
         mllin.product_matrix_vector(self.Ws[h],self.layers[h],self.layer_acts[h+1])
         self.layer_acts[h+1] += self.cs[h]
         mlnonlin.sigmoid(self.layer_acts[h+1],self.layers[h+1])

      mllin.product_matrix_vector(self.U,self.layers[-1],self.output_act)
      self.output_act += self.d
      mlnonlin.softmax(self.output_act,output)

      return [output.argmax(),output]

   def cost(self,outputs,example):
      target = example[1]
      class_id,output = outputs

      return [ target!=class_id,-np.log(output[target])]

   def verify_gradients(self):
      
      print 'WARNING: calling verify_gradients reinitializes the learner'

      rng = np.random.mtrand.RandomState(1234)
      input_order = range(20)
      rng.shuffle(input_order)

      self.seed = 1234
      self.hidden_sizes = [4,5,6]
      self.initialize_learner({'input_size':20,'targets':set([0,1,2])})
      example = (rng.rand(20)<0.5,2)
      epsilon=1e-6
      self.learning_rate = 1
      self.decrease_constant = 0

      import copy
      Ws_copy = copy.deepcopy(self.Ws)
      emp_dWs = copy.deepcopy(self.Ws)
      for h in range(self.n_hidden_layers):
         for i in range(self.Ws[h].shape[0]):
            for j in range(self.Ws[h].shape[1]):
               self.Ws[h][i,j] += epsilon
               output = self.use_learner(example)
               a = self.cost(output,example)[1]
               self.Ws[h][i,j] -= epsilon
               
               self.Ws[h][i,j] -= epsilon
               output = self.use_learner(example)
               b = self.cost(output,example)[1]
               self.Ws[h][i,j] += epsilon

               emp_dWs[h][i,j] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.Ws = Ws_copy
      print 'dWs[0] diff.:',np.sum(np.abs(self.dWs[0].ravel()-emp_dWs[0].ravel()))/self.Ws[0].ravel().shape[0]
      print 'dWs[1] diff.:',np.sum(np.abs(self.dWs[1].ravel()-emp_dWs[1].ravel()))/self.Ws[1].ravel().shape[0]
      print 'dWs[2] diff.:',np.sum(np.abs(self.dWs[2].ravel()-emp_dWs[2].ravel()))/self.Ws[2].ravel().shape[0]

      cs_copy = copy.deepcopy(self.cs)
      emp_dcs = copy.deepcopy(self.cs)
      for h in range(self.n_hidden_layers):
         for i in range(self.cs[h].shape[0]):
            self.cs[h][i] += epsilon
            output = self.use_learner(example)
            a = self.cost(output,example)[1]
            self.cs[h][i] -= epsilon
            
            self.cs[h][i] -= epsilon
            output = self.use_learner(example)
            b = self.cost(output,example)[1]
            self.cs[h][i] += epsilon
            
            emp_dcs[h][i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.cs = cs_copy
      print 'dcs[0] diff.:',np.sum(np.abs(self.dcs[0].ravel()-emp_dcs[0].ravel()))/self.cs[0].ravel().shape[0]
      print 'dcs[1] diff.:',np.sum(np.abs(self.dcs[1].ravel()-emp_dcs[1].ravel()))/self.cs[1].ravel().shape[0]
      print 'dcs[2] diff.:',np.sum(np.abs(self.dcs[2].ravel()-emp_dcs[2].ravel()))/self.cs[2].ravel().shape[0]

      U_copy = np.array(self.U)
      emp_dU = np.zeros(self.U.shape)
      for i in range(self.U.shape[0]):
         for j in range(self.U.shape[1]):
            self.U[i,j] += epsilon
            output = self.use_learner(example)
            a = self.cost(output,example)[1]
            self.U[i,j] -= epsilon

            self.U[i,j] -= epsilon
            output = self.use_learner(example)
            b = self.cost(output,example)[1]
            self.U[i,j] += epsilon

            emp_dU[i,j] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.U[:] = U_copy
      print 'dU diff.:',np.sum(np.abs(self.dU.ravel()-emp_dU.ravel()))/self.U.ravel().shape[0]

      d_copy = np.array(self.d)
      emp_dd = np.zeros(self.d.shape)
      for i in range(self.d.shape[0]):
         self.d[i] += epsilon
         output = self.use_learner(example)
         a = self.cost(output,example)[1]
         self.d[i] -= epsilon
         
         self.d[i] -= epsilon
         output = self.use_learner(example)
         b = self.cost(output,example)[1]
         self.d[i] += epsilon
         
         emp_dd[i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.d[:] = d_copy
      print 'dd diff.:',np.sum(np.abs(self.dd.ravel()-emp_dd.ravel()))/self.d.ravel().shape[0]
