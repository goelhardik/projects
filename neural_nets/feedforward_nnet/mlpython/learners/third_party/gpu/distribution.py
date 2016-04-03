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
The ``learners.third_party.gpu.classification`` module contains 
density or distribution estimation algorithms that uses a GPU.

* RestrictedBoltzmannMachine:      A Restricted Boltzmann Machine (RBM) distribution estimator, on the GPU.

"""

from mlpython.learners.generic import Learner
import numpy as np
try :
    import cudamat as cm
except ImportError:
    import warnings
    warnings.warn('\'import cudamat\' failed. The CUDAMat library is not properly installed. See http://code.google.com/p/cudamat/ for instructions.')
except OSError:
    import warnings
    warnings.warn('\'import cudamat\' failed. The CUDAMat library is not properly installed. See http://code.google.com/p/cudamat/ for instructions.')


class RestrictedBoltzmannMachine(Learner):
    """
    A Restricted Boltzmann Machine (RBM) distribution estimator, on the GPU.

    Given an input, the RBM will assign it a score,
    corresponding to its negative free-energy. This score 
    corresponds to the RBM unnormalized log-likelihood.

    Most options are self-explanatory. The frequency at which data
    is obtained from the training set and loaded onto the GPU
    is controlled by option load_data_every. It specifies
    the amount of data to be loaded in number of minibatches. 
    If < 1, then data is loaded just once.

    Options:
    - 'n_stages'
    - 'latent_size'
    - 'learning_rate'
    - 'momentum'
    - 'n_gibbs_steps'
    - 'use_persistent_chain'
    - 'minibatch_size'
    - 'load_data_every'

    Required metadata:
    - 'input_size'

    """
    def __init__(   self,
                    n_stages=10,                # number of training iterations
                    latent_size=100,            # hidden layer size
                    learning_rate=1e-2,         # learning rate
                    momentum=0,                 # momentum
                    n_gibbs_steps=1,            # number of Gibbs sampling steps
                    use_persistent_chain=False, # use persistent CD?
                    minibatch_size=128,         # size of minibatch
                    load_data_every=-1,         # how frequently to load data to GPU
                    seed=1234
                    ):
        self.stage = 0
        self.reload_data = True
        self.n_stages = n_stages
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_gibbs_steps = n_gibbs_steps
        self.use_persistent_chain = use_persistent_chain
        self.minibatch_size = minibatch_size
        self.load_data_every = load_data_every
        self.seed = seed
        self.gpu_dataset = None

        cm.cuda_set_device(0)
        cm.init()

        self.rng = np.random.mtrand.RandomState(seed)
        cm.CUDAMatrix.init_random(seed = seed)

    def __del__(self):
        cm.shutdown()

    def train(self,trainset):
        """
        Trains the RBM on the GPU
        """

        if self.stage == 0:
            self.input_size = trainset.metadata['input_size']
            self.forget()

        if self.minibatch_size > len(trainset):
            print 'Warning: minibatch_size is larger than training set.'
            print '         Setting minibatch_szie to size of training set...'

        if self.load_data_every*self.minibatch_size >= len(trainset):
            # data fits in one load, so load the data once
            self.load_data_every = -1

        # Preparing training...
        if self.load_data_every < 1 and self.reload_data:
            if self.reload_data:
                if self.gpu_dataset != None:
                    self.gpu_dataset.free_device_memory()
                self.gpu_dataset = cm.empty((self.input_size,len(trainset)))
                self.gpu_dataset.copy_to_host()

                # load data to GPU
                for input,t in zip(trainset,range(len(trainset))):
                    self.gpu_dataset.numpy_array[:,t] = input.T
                        
                self.gpu_dataset.copy_to_device()
                self.reload_data = False
        else:
            n_loaded = 0
            if self.gpu_dataset == None or self.gpu_dataset.shape != (self.input_size,self.load_data_every*self.minibatch_size):
                if self.gpu_dataset != None:
                    self.gpu_dataset.free_device_memory()
                self.gpu_dataset = cm.empty((self.input_size,
                                             self.load_data_every*self.minibatch_size)) 
                self.gpu_dataset.copy_to_host()

        while self.stage < self.n_stages:
            err = 0.
            count = 0
            if self.load_data_every < 1:  # Is the whole dataset loaded...
                err += self.train_on_loaded_data(len(trainset))
                count += 1
            else:                         # ... otherwise load it as you go.
                for input in trainset:
                    # load some data on GPU
                    self.gpu_dataset.numpy_array[:,n_loaded] = input.T
                    n_loaded += 1
                    if n_loaded >= self.load_data_every*self.minibatch_size:
                        self.gpu_dataset.copy_to_device()
                        err += self.train_on_loaded_data(n_loaded)
                        count += 1
                        n_loaded = 0
                
                if n_loaded > 0:
                    # Train on last portion of data
                    self.gpu_dataset.copy_to_device()
                    n_loaded = max(n_loaded,self.minibatch_size) # ensure enough data for one minibatch
                    err += self.train_on_loaded_data(n_loaded)
                    count += 1
                    n_loaded = 0

            self.stage += 1
            print 'Average mini-batch reconstruction error:',err/count

    def train_on_loaded_data(self,n_loaded):
        """
        Trains on data already loaded on GPU. 
        There most be enough data for at least one mini-batch.
        """
        err = 0.
        count = 0
        #self.print_first_row = True
        # Update from first minibatch to previous-to-last one
        for t in range(self.minibatch_size,n_loaded,self.minibatch_size):
            err += self.rbm_update(self.gpu_dataset.slice(t-self.minibatch_size,t))
            count += 1
        #    self.print_first_row = False

        # special case: last minibatch (which might be smaller)
        err += self.rbm_update(self.gpu_dataset.slice(
                n_loaded-self.minibatch_size,n_loaded))
        count += 1
        return err/count

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.rng = np.random.mtrand.RandomState(self.seed)
        cm.CUDAMatrix.init_random(seed = self.seed)
        self.reload_data = True

        # Initializing parameters
        self.W = cm.CUDAMatrix(self.rng.randn(self.latent_size,self.input_size)/float(self.input_size))
        self.dW = cm.CUDAMatrix(np.zeros((self.latent_size,self.input_size)))
        self.c = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.dc = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.b = cm.CUDAMatrix(np.zeros((self.input_size,1)))
        self.db = cm.CUDAMatrix(np.zeros((self.input_size,1)))

        # Create data memory allocation (column-wise)
        self.gpu_x = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size)))
        self.gpu_x_sample = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size)))
        self.gpu_h = cm.CUDAMatrix(np.zeros((self.latent_size,self.minibatch_size)))
        self.gpu_h_sample = cm.CUDAMatrix(np.zeros((self.latent_size,self.minibatch_size)))

        self.gpu_negative_free_energy = cm.CUDAMatrix(np.zeros((1,self.minibatch_size)))

    def rbm_update(self,gpu_data):

        # Positive phase
        cm.dot(self.W,gpu_data,self.gpu_h)
        self.gpu_h.add_col_vec(self.c)
        self.gpu_h.apply_sigmoid()

        self.dW.mult(self.momentum)
        self.dc.mult(self.momentum)
        self.db.mult(self.momentum)
        self.dW.add_dot(self.gpu_h,gpu_data.T)
        self.dc.add_sums(self.gpu_h,axis=1,mult=1.)
        self.db.add_sums(gpu_data,axis=1,mult=1.)

        if self.use_persistent_chain:
            cm.dot(self.W,self.gpu_x_sample,self.gpu_h)
            self.gpu_h.add_col_vec(self.c)
            self.gpu_h.apply_sigmoid()

        for it in range(self.n_gibbs_steps):
            self.gpu_h_sample.fill_with_rand()
            self.gpu_h_sample.less_than(self.gpu_h)

            # Down pass
            cm.dot(self.W.T,self.gpu_h_sample,self.gpu_x)
            self.gpu_x.add_col_vec(self.b)
            self.gpu_x.apply_sigmoid()
            self.gpu_x_sample.fill_with_rand()
            self.gpu_x_sample.less_than(self.gpu_x)

            # Up pass
            cm.dot(self.W,self.gpu_x_sample,self.gpu_h)
            self.gpu_h.add_col_vec(self.c)
            self.gpu_h.apply_sigmoid()
        
        self.dW.subtract_dot(self.gpu_h,self.gpu_x_sample.T)
        self.dc.add_sums(self.gpu_h,axis=1,mult=-1.)
        self.db.add_sums(self.gpu_x_sample,axis=1,mult=-1.)

        # Update RBM
        self.W.add_mult(self.dW,alpha=self.learning_rate/self.minibatch_size)
        self.c.add_mult(self.dc,alpha=self.learning_rate/self.minibatch_size)
        self.b.add_mult(self.db,alpha=self.learning_rate/self.minibatch_size)

        #if self.print_first_row:
        #    gpu_data.copy_to_host()
        #    print gpu_data.numpy_array[:,0]
        #    self.gpu_x.copy_to_host()
        #    print self.gpu_x.numpy_array[:,0]

        # Compute reconstruction error
        self.gpu_x.subtract(gpu_data)
        err = self.gpu_x.euclid_norm()
        err = err**2
        err /= self.gpu_x.shape[1]
        return err

    def negative_free_energy(self,gpu_data):
        """
        Computes the negative free-energy.
        Outputs a reference to a pre-allocated GPU variable
        containing the result.
        """

        cm.dot(self.W,gpu_data,self.gpu_h)
        self.gpu_h.add_col_vec(self.c)
        # to avoid memory creation, using gpu_h
        # and gpu_h_sample for these computations
        cm.exp(self.gpu_h,self.gpu_h_sample)
        self.gpu_h_sample.add(1.)
        cm.log(self.gpu_h_sample,self.gpu_h)
        self.gpu_h.sum(axis=0,target=self.gpu_negative_free_energy)
        self.gpu_negative_free_energy.add_dot(self.b.T,gpu_data)
        return self.gpu_negative_free_energy

    def use(self,dataset):
        """
        Outputs the negative free-energy.
        """
        outputs = np.zeros((len(dataset),1))
        n_loaded = 0
        self.gpu_x.copy_to_host()
        t = 0
        for input in dataset:
        #for input,t in zip(dataset,range(len(dataset))):
            # load some data on GPU
            self.gpu_x.numpy_array[:,n_loaded] = input.T
            n_loaded += 1
            if n_loaded >= self.minibatch_size:
                self.gpu_x.copy_to_device()
                nfe = self.negative_free_energy(self.gpu_x)
                nfe.copy_to_host()
                outputs[(t+1-self.minibatch_size):(t+1),:] = nfe.numpy_array.T
                n_loaded = 0
            t += 1

        # Compute for last examples!
        if n_loaded > 0:
            self.gpu_x.copy_to_device()
            nfe = self.negative_free_energy(self.gpu_x)
            nfe.copy_to_host()
            outputs[(len(dataset)-n_loaded):,:] = nfe.numpy_array[:,:n_loaded].T
            
        return outputs

    def test(self,dataset):
        """
        Outputs the NLLs of each example.
        """
        outputs = self.use(dataset)
        costs = zeros(len(dataset),1)
        for o,c in zip(outputs,costs):
            c[0] = -o[0]

        return outputs,costs
