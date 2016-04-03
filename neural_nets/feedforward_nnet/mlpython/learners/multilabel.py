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
The ``learners.multilabel`` module contains Learners meant for multi-label classification. 
The MLProblems for these Learners should be iterators over (input,target) pairs, where
the target is a vector of binary labels. Metadata keyword 'target_size', corresponding
to the size of the target vector, will typically be required.

The currently implemented algorithms are:

* MultilabelCRF: a Conditional Random Field for multi-label classification

"""

import mlpython.misc.io as mlio
import mlpython.mathutils.nonlinear as mlnonlin
from mlpython.learners.generic import Learner 
import numpy as np

class MultilabelCRF(Learner):
    """
    Conditional Random Field for multi-label classification
    
    Option ``n_stages`` determines the number of iterations
    over the training set when ``train()``is called. 

    Option ``lr`` is the learning rate (default=0.001).

    Option ``approximate_inference`` is a string describing the
    approximate inference method to use at test time. If equal to
    ``'mean_field'`` (default), will use mean-field inference. If equal to
    ``loopy_belief_propagation``, will use loopy belief propagation.
    This is essentially the number of times messages are being passed
    across the pairwise connection matrix between the labels. If
    is set to 0, the model corresponds to multiple logistic regressors.

    Option ``n_inference_iterations`` is the number of iterations for
    approximate inference (default=5).

    Option ``damping_factor`` is the damping factor to be used by
    loopy belief propagation (default=0).

    Option ``binary_outputs`` determines whether the predicted outputs
    (target marginals) should be binarized using a threshold at 0.5
    (default=True).

    Option ``seed`` is the number seed of the internal random number
    generator (default=1234).

    **Required metadata:**

    * ``'input_size'``: size of the inputs
    * ``'target_size'``: size of the targets

    """
    
    def __init__(self, n_stages, lr=0.001, approximate_inference='mean_field',
                 n_inference_iterations=5, damping_factor=0, binary_outputs=True,
                 seed=1234):
    
        self.n_stages = n_stages
        self.lr = lr
        self.approximate_inference = approximate_inference
        self.n_inference_iterations = n_inference_iterations
        self.damping_factor = damping_factor
        self.seed = seed
        self.binary_outputs = binary_outputs
        self.stage = 0

    def initialize(self,trainset):
        """
        Initializes training.
        """

        # Initializes RBM parameters
        self.rng = np.random.mtrand.RandomState(self.seed)
        self.input_size = trainset.metadata['input_size']
        self.target_size = trainset.metadata['target_size']

        self.V = (2*self.rng.rand(self.input_size,self.target_size)-1)/self.input_size
        self.U = (2*self.rng.rand(self.target_size,self.target_size)-1)/self.target_size
        self.U = 0.5*(self.U+self.U.T)
        self.U -= np.diag(np.diag(self.U))
        self.b = np.zeros((self.target_size))

    def train(self,trainset):
        if self.stage == 0:
            self.initialize(trainset)

        for it in range(self.stage,self.n_stages):
            for example in trainset:
                input,target = example
                if len(input.shape) == 1:
                    input = input.reshape((1,-1))
                    target = target.reshape((1,-1))

                data_vis_bias = np.dot(input,self.V)+self.b
                
                if self.n_inference_iterations > 0:
                    if self.approximate_inference == 'mean_field':
                        target_singles = 1./(1+np.exp(-data_vis_bias))
                        for k in range(self.n_inference_iterations):
                            if self.damping_factor > 0:
                                target_singles *= self.damping_factor
                                target_singles += (1.-self.damping_factor)/(1+np.exp(-data_vis_bias-np.dot(target_singles,self.U)))
                            else:
                                target_singles = 1./(1+np.exp(-data_vis_bias-np.dot(target_singles,self.U)))                                
                        sum_target_pairs = np.dot(target_singles.T,target_singles)
                    elif self.approximate_inference == 'loopy_belief_propagation':
                        target_singles = np.zeros((input.shape[0],self.target_size))
                        sum_target_pairs = np.zeros((self.target_size,self.target_size))

                        log_messages = np.zeros((self.target_size,self.target_size))
                        softplus_out1 = np.zeros((self.target_size,self.target_size))
                        softplus_out2 = np.zeros((self.target_size,self.target_size))
                        for i in range(input.shape[0]):
                            # Loopy BP
                            data_bias = data_vis_bias[i]
                            log_messages[:] = 0
                            for k in range(self.n_inference_iterations):
                                # Update messages
                                acc_log_messages = data_bias + log_messages.sum(axis=1) - log_messages
                                if self.damping_factor > 0:
                                    log_messages *= self.damping_factor
                                    mlnonlin.softplus(acc_log_messages,softplus_out2)
                                    acc_log_messages += self.U
                                    mlnonlin.softplus(acc_log_messages,softplus_out1)
                                    log_messages += (1-self.damping_factor) * (softplus_out1-softplus_out2)
                                else:
                                    mlnonlin.softplus(acc_log_messages,softplus_out2)
                                    acc_log_messages += self.U
                                    mlnonlin.softplus(acc_log_messages,softplus_out1)
                                    log_messages = softplus_out1-softplus_out2

                            # Single marginals
                            target_singles[i] = 1./(1+np.exp(-data_bias-log_messages.sum(axis=1)))
                            # Pair-wise marginals
                            acc_log_messages = data_bias + log_messages.sum(axis=1) - log_messages
                            p11 = np.exp(self.U+acc_log_messages.T+acc_log_messages)
                            p10 = np.exp(acc_log_messages)
                            p01 = np.exp(acc_log_messages.T)
                            sum_p = 1 + p11 + p10 + p01
                            sum_target_pairs += p11 / sum_p
                    else:
                        raise ValueError('approximate_inference \'%\' unknown' % self.approximate_inference)

                else:
                    target_singles = 1./(1+np.exp(-data_vis_bias))

                # apply CRF gradient update
                db = np.sum(target_singles,axis=0) - np.sum(target,axis=0)
                dV = np.dot(input.T,target_singles-target)
                self.b -= self.lr/input.shape[0] * db
                self.V -= self.lr/input.shape[0] * dV

                if self.n_inference_iterations > 0:
                    dU = sum_target_pairs - np.dot(target.T,target)
                    self.U -= self.lr/input.shape[0] * dU 
                    # Ensuring symmetry and 0 diagonal
                    self.U = 0.5*(self.U+self.U.T)
                    self.U -= np.diag(np.diag(self.U))

        self.stage = self.n_stages

    def forget(self):
        self.stage = 0
        self.V = (2*self.rng.rand(self.input_size,self.target_size)-1)/self.input_size
        self.U = (2*self.rng.rand(self.target_size,self.target_size)-1)/self.target_size
        self.U = 0.5*(self.U+self.U.T)
        self.U -= np.diag(np.diag(self.U))
        self.b = np.zeros((self.target_size))

    def use(self,dataset):
        outputs = []
        for example in dataset:
            input,dummy = example
            if len(input.shape) == 1:
                input = input.reshape((1,-1))

            data_vis_bias = np.dot(input,self.V)+self.b
            if self.n_inference_iterations > 0:
                if self.approximate_inference == 'mean_field':
                    target_singles = 1./(1+np.exp(-data_vis_bias))
                    for k in range(self.n_inference_iterations):
                        if self.damping_factor > 0:
                            target_singles *= self.damping_factor
                            target_singles += (1.-self.damping_factor)/(1+np.exp(-data_vis_bias-np.dot(target_singles,self.U)))
                        else:
                            target_singles = 1./(1+np.exp(-data_vis_bias-np.dot(target_singles,self.U)))                                
                elif self.approximate_inference == 'loopy_belief_propagation':
                    target_singles = np.zeros((input.shape[0],self.target_size))

                    log_messages = np.zeros((self.target_size,self.target_size))
                    softplus_out1 = np.zeros((self.target_size,self.target_size))
                    softplus_out2 = np.zeros((self.target_size,self.target_size))
                    for i in range(input.shape[0]):
                        # Loopy BP
                        data_bias = data_vis_bias[i]
                        log_messages[:] = 0
                        for k in range(self.n_inference_iterations):
                            # Update messages
                            acc_log_messages = data_bias + log_messages.sum(axis=1) - log_messages
                            if self.damping_factor > 0:
                                log_messages *= self.damping_factor
                                mlnonlin.softplus(acc_log_messages,softplus_out2)
                                acc_log_messages += self.U
                                mlnonlin.softplus(acc_log_messages,softplus_out1)
                                log_messages += (1-self.damping_factor) * (softplus_out1-softplus_out2)
                            else:
                                mlnonlin.softplus(acc_log_messages,softplus_out2)
                                acc_log_messages += self.U
                                mlnonlin.softplus(acc_log_messages,softplus_out1)
                                log_messages = softplus_out1-softplus_out2

                        # Single marginals
                        target_singles[i] = 1./(1+np.exp(-data_bias-log_messages.sum(axis=1)))
                else:
                    raise ValueError('approximate_inference \'%\' unknown' % self.approximate_inference)

            else:
                target_singles = 1./(1+np.exp(-data_vis_bias))

            if self.binary_outputs:
                outputs += [ target_singles > 0.5 ]
            else: 
                outputs += [ target_singles ]

        return outputs
            
    def test(self,dataset):
        outputs = self.use(dataset)
        costs = []
        for example,output in zip(dataset,outputs):
            dummy,target = example
            costs += [ np.mean((target - output)**2,axis=0) ]

        return outputs,costs
