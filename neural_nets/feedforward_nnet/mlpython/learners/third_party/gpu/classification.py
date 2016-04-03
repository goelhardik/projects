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
classification algorithms that uses a GPU.

* ClassificationRestrictedBoltzmannMachine:      A Classification Restricted Boltzmann Machine on the GPU.

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


class ClassificationRestrictedBoltzmannMachine(Learner):
    """
    A Classification Restricted Boltzmann Machine on the GPU.

    This learner supports semi-supervised learning, i.e.
    will take advantage of unlabeled examples in the
    training set. Such examples should have None as
    a target.

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
    - 'gen_learning_weight'
    - 'semisup_learning_weight'
    - 'n_gibbs_steps'
    - 'use_persistent_chain'
    - 'minibatch_size'
    - 'load_data_every'

    Required metadata:
    - 'input_size'
    - 'targets'

    Reference: Classification using Discriminative Restricted Boltzmann Machines
               Larochelle and Bengio
               http://icml2008.cs.helsinki.fi/papers/601.pdf

    Notes:
    - for simplicity, instead of sampling from the softmax class units, the code uses
      the softmax probabilities directly (easier with the GPU)

    """
    def __init__(   self,
                    n_stages=10,                  # number of training iterations
                    latent_size=100,              # hidden layer size
                    learning_rate=1e-2,           # learning rate
                    momentum=0,                   # momentum
                    gen_learning_weight=0.01,     # generative learning weight
                    semisup_learning_weight=0.01, # semi-supervised learning weight
                    n_gibbs_steps=1,              # number of Gibbs sampling steps
                    use_persistent_chain=False,   # use persistent CD?
                    minibatch_size=128,           # size of minibatch
                    load_data_every=-1,           # how frequently to load data to GPU
                    seed=1234
                    ):
        self.stage = 0
        self.reload_data = True
        self.n_stages = n_stages
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gen_learning_weight = gen_learning_weight
        self.semisup_learning_weight = semisup_learning_weight
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
            self.n_classes = len(trainset.metadata['targets'])
            self.forget()

        if self.minibatch_size > len(trainset):
            print 'Warning: minibatch_size is larger than training set.'
            print '         Setting minibatch_size to size of training set...'

        if self.load_data_every*self.minibatch_size >= len(trainset):
            # data fits in one load, so load the data once
            self.load_data_every = -1

        ## Use this if wants to support fancy target representations
        # Creating target vectors, on the GPU
        self.gpu_target_vectors = cm.empty((self.n_classes,self.n_classes))
        self.gpu_target_vectors.copy_to_host()
        self.gpu_target_vectors.numpy_array[:] = 0
        self.target_vectors = np.zeros((self.n_classes,self.n_classes))
        for c in range(self.n_classes):
            self.gpu_target_vectors.numpy_array[c,c] = 1
            self.target_vectors[c,c] = 1
        self.gpu_target_vectors.copy_to_device()

        # Preparing training...
        if self.load_data_every < 1:
            if self.reload_data:
                if self.gpu_dataset != None:
                    self.gpu_dataset.free_device_memory()
                self.gpu_dataset = cm.empty((self.input_size,len(trainset)))
                self.gpu_dataset.copy_to_host()
                self.dataset_targets = np.zeros((len(trainset)),dtype=int)

                # load data to GPU
                self.gpu_dataset.numpy_array[:] = 0
                for t,example in enumerate(trainset):
                    input,target = example
                    self.gpu_dataset.numpy_array[:,t] = input
                    if target != None:  self.dataset_targets[t] = target
                    else:               self.dataset_targets[t] = -1

                self.gpu_dataset.copy_to_device()
                self.reload_data = False
        else:
            n_loaded = 0
            if self.gpu_dataset == None or self.gpu_dataset.shape != (self.input_size,self.load_data_every*self.minibatch_size):
                if self.gpu_dataset != None:
                    self.gpu_dataset.free_device_memory()
                self.gpu_dataset = cm.empty((self.input_size,
                                             self.load_data_every*self.minibatch_size)) 
                self.dataset_targets = np.zeros((self.load_data_every*self.minibatch_size),
                                                dtype=int)
                self.gpu_dataset.copy_to_host()

        while self.stage < self.n_stages:
            if self.load_data_every < 1:  # Is the whole dataset loaded...
                self.train_on_loaded_data()
            else:                         # ... otherwise load it as you go.
                for example in trainset:
                    input,target = example
                    # load some data on GPU
                    self.gpu_dataset.numpy_array[:,n_loaded] = input.T
                    if target != None:   self.dataset_targets[n_loaded] = target
                    else:                self.dataset_targets[n_loaded] = -1
                    n_loaded += 1
                    if n_loaded >= self.load_data_every*self.minibatch_size:
                        self.gpu_dataset.copy_to_device()
                        self.train_on_loaded_data()
                        n_loaded = 0
                
                if n_loaded > 0:
                    # Train on last portion of data
                    self.gpu_dataset.copy_to_device()
                    self.train_on_loaded_data()
                    n_loaded = 0

            self.stage += 1

    def train_on_loaded_data(self):
        """
        Trains on data already loaded on GPU. 
        """
        n_labeled = 0
        n_unlabeled = 0

        #self.print_first_row = True
        for t,target in enumerate(self.dataset_targets):
            if target == -1:
                self.gpu_unlabeled_x.slice(n_unlabeled,n_unlabeled+1).assign(
                    self.gpu_dataset.slice(t,t+1))
                n_unlabeled += 1
            else:
                self.gpu_labeled_x.slice(n_labeled,n_labeled+1).assign(
                    self.gpu_dataset.slice(t,t+1))
                self.gpu_target_for_x.numpy_array[:,n_labeled] = 0
                self.gpu_target_for_x.numpy_array[target,n_labeled] = 1
                n_labeled += 1
            
            if n_labeled == self.minibatch_size:
                self.gpu_target_for_x.copy_to_device()
                self.rbm_update(self.gpu_labeled_x,self.gpu_target_for_x)
                n_labeled = 0
            if n_unlabeled == self.minibatch_size:
                self.rbm_update(self.gpu_unlabeled_x)
                n_unlabeled = 0

        if n_labeled > 0:
            self.rbm_update(self.gpu_labeled_x,self.gpu_target_for_x,n_first_update=n_labeled)
        if n_unlabeled > 0:
            self.rbm_update(self.gpu_unlabeled_x,n_first_update=n_unlabeled)

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.rng = np.random.mtrand.RandomState(self.seed)
        cm.CUDAMatrix.init_random(seed = self.seed)
        self.reload_data = True

        # Initializing parameters
        self.W = cm.CUDAMatrix(self.rng.randn(self.latent_size,self.input_size)/float(self.input_size))
        self.dW = cm.CUDAMatrix(np.zeros((self.latent_size,self.input_size)))
        self.U = cm.CUDAMatrix(self.rng.randn(self.latent_size,self.n_classes)/float(self.latent_size))
        self.dU = cm.CUDAMatrix(np.zeros((self.latent_size,self.n_classes)))
        self.c = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.dc = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.b = cm.CUDAMatrix(np.zeros((self.input_size,1)))
        self.db = cm.CUDAMatrix(np.zeros((self.input_size,1)))
        self.d = cm.CUDAMatrix(np.zeros((self.n_classes,1)))
        self.dd = cm.CUDAMatrix(np.zeros((self.n_classes,1)))

        # Create data memory allocation (column-wise)
        self.gpu_x = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size)))
        self.gpu_x_sample = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size)))
        self.gpu_x_persistent = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size))) #PCD
        self.gpu_y = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))
        self.gpu_y_persistent = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))  #PCD
        self.gpu_h = cm.CUDAMatrix(np.zeros((self.latent_size,self.minibatch_size)))
        self.gpu_h_sample = cm.CUDAMatrix(np.zeros((self.latent_size,self.minibatch_size)))
        self.gpu_target_vec_pos = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))
        self.gpu_target_vec_neg = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))

        # Used for softmax computations during CD
        self.gpu_y_trans = cm.CUDAMatrix(np.zeros((self.minibatch_size,self.n_classes)))
        self.gpu_y_trans_mean = cm.CUDAMatrix(np.zeros((self.minibatch_size,1)))
        self.gpu_y_trans_norm = cm.CUDAMatrix(np.zeros((self.minibatch_size,1)))

        # Containers for data on which to do an update
        self.gpu_labeled_x = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size)))
        self.gpu_unlabeled_x = cm.CUDAMatrix(np.zeros((self.input_size,self.minibatch_size)))
        self.gpu_target_for_x = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))
        self.gpu_target_for_x.copy_to_device()

        # Temporary computations variables
        self.gpu_act_from_x = cm.CUDAMatrix(np.zeros((self.latent_size,self.minibatch_size)))
        self.gpu_act_from_y = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.gpu_bias_from_y = cm.CUDAMatrix(np.zeros((1,1)))
        self.gpu_p_y_given_x = cm.CUDAMatrix(np.zeros((self.minibatch_size,self.n_classes)))
        self.gpu_p_y_given_x_norm = cm.CUDAMatrix(np.zeros((self.minibatch_size,1)))
        self.gpu_p_y_given_x_trans = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))
        self.gpu_negative_free_energy_for_y = cm.CUDAMatrix(np.zeros((1,self.minibatch_size)))
        self.gpu_negative_free_energy = cm.CUDAMatrix(np.zeros((self.minibatch_size,self.n_classes)))

        self.gpu_mean_negative_free_energy = cm.CUDAMatrix(np.zeros((self.minibatch_size,1)))
        self.gpu_dhidact = cm.CUDAMatrix(np.zeros((self.latent_size,self.minibatch_size)))
        self.gpu_dhidact_sum = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.gpu_doutput = cm.CUDAMatrix(np.zeros((self.n_classes,self.minibatch_size)))
        self.gpu_doutput_sum = cm.CUDAMatrix(np.zeros((self.n_classes,1)))
        self.gpu_doutput_row = cm.CUDAMatrix(np.zeros((1,self.minibatch_size)))
        self.gpu_doutput_trans = cm.CUDAMatrix(np.zeros((self.minibatch_size,self.n_classes)))

        
        # CD stats
        self.cd_W = cm.CUDAMatrix(np.zeros((self.latent_size,self.input_size)))
        self.cd_U = cm.CUDAMatrix(np.zeros((self.latent_size,self.n_classes)))
        self.cd_c = cm.CUDAMatrix(np.zeros((self.latent_size,1)))
        self.cd_b = cm.CUDAMatrix(np.zeros((self.input_size,1)))
        self.cd_d = cm.CUDAMatrix(np.zeros((self.n_classes,1)))


    def compute_output(self,gpu_data):
        """
        Computes p(y|x). Puts the result in self.gpu_p_y_given_x.
        """
        
        cm.dot(self.W,gpu_data,self.gpu_act_from_x)
        self.gpu_act_from_x.add_col_vec(self.c)
        for c in range(self.n_classes):
            cm.dot(self.U,self.gpu_target_vectors.slice(c,c+1),self.gpu_act_from_y)
            # to avoid memory creation, using gpu_h
            # and gpu_h_sample for these computations
            self.gpu_act_from_x.add_col_vec(self.gpu_act_from_y,target=self.gpu_h)
            cm.exp(self.gpu_h,self.gpu_h_sample)
            self.gpu_h_sample.add(1.)
            cm.log(self.gpu_h_sample,self.gpu_h)
            self.gpu_h.sum(axis=0,target=self.gpu_negative_free_energy_for_y)
            cm.dot(self.d.T,self.gpu_target_vectors.slice(c,c+1),target=self.gpu_bias_from_y)
            self.gpu_negative_free_energy_for_y.add_col_vec(self.gpu_bias_from_y)
            self.gpu_negative_free_energy_for_y.transpose(target=self.gpu_negative_free_energy.slice(c,c+1))
        # Subtracting mean for more stable softmax computation
        self.gpu_negative_free_energy.sum(axis=1,target=self.gpu_mean_negative_free_energy)
        self.gpu_mean_negative_free_energy.divide(-self.n_classes)
        self.gpu_negative_free_energy.add_col_vec(self.gpu_mean_negative_free_energy)

        cm.exp(self.gpu_negative_free_energy,target=self.gpu_negative_free_energy)
        self.gpu_negative_free_energy.sum(axis=1,target=self.gpu_p_y_given_x_norm)
        for c in range(self.n_classes):
            self.gpu_negative_free_energy.slice(c,c+1).divide(self.gpu_p_y_given_x_norm,
                                                              target=self.gpu_p_y_given_x.slice(c,c+1))
        self.gpu_p_y_given_x.transpose(target=self.gpu_p_y_given_x_trans)

    def rbm_update(self,gpu_data,gpu_target_for_data=None,n_first_update=None):

        is_labeled = gpu_target_for_data != None

        if n_first_update != None:
            gpu_data.slice(n_first_update,self.minibatch_size).assign(0)

        self.dW.mult(self.momentum)
        self.dU.mult(self.momentum)
        self.dc.mult(self.momentum)
        self.db.mult(self.momentum)
        self.dd.mult(self.momentum)

        # Computes p(y|x). This methods fills in
        # self.gpu_p_y_given_x and self.gpu_p_y_given_x_trans with the result.
        # It also computes self.gpu_act_from_x.
        self.compute_output(gpu_data)

        if gpu_target_for_data != None:
            # Compute discriminative gradient
            self.gpu_p_y_given_x_trans.subtract(gpu_target_for_data,self.gpu_doutput)
            if n_first_update != None:
                # Making sure gradient is non-zero only for n_first_update first examples
                self.gpu_doutput.slice(n_first_update,self.minibatch_size).assign(0)

            self.gpu_doutput.sum(axis=1,target=self.gpu_doutput_sum)
            self.dd.add_dot(self.gpu_target_vectors,self.gpu_doutput_sum)
            self.gpu_dhidact.assign(0)
            self.gpu_doutput.transpose(self.gpu_doutput_trans)
            for c in range(self.n_classes):
                cm.dot(self.U,self.gpu_target_vectors.slice(c,c+1),self.gpu_act_from_y)
                # to avoid memory creation, using gpu_h
                # and gpu_h_sample for these computations
                self.gpu_act_from_x.add_col_vec(self.gpu_act_from_y,target=self.gpu_h)
                self.gpu_h.apply_sigmoid()
                self.gpu_doutput_trans.slice(c,c+1).transpose(target=self.gpu_doutput_row)
                self.gpu_h.mult_by_row(self.gpu_doutput_row)
                self.gpu_dhidact.add(self.gpu_h)
                self.dc.add_sums(self.gpu_h,axis=1)
                self.gpu_h.sum(axis=1,target=self.gpu_dhidact_sum)
                self.dU.add_dot(self.gpu_dhidact_sum,self.gpu_target_vectors.slice(c,c+1).T)

            self.dW.add_dot(self.gpu_dhidact,gpu_data.T)

        else:
            # Sample a y according to p(y|x)
            # ... actually, we use the softmax probs, it's much simpler
            gpu_target_for_data = self.gpu_p_y_given_x_trans

        if (is_labeled and self.gen_learning_weight > 0) or (not is_labeled and self.semisup_learning_weight > 0):

            self.cd_W.assign(0)
            self.cd_U.assign(0)
            self.cd_c.assign(0)
            self.cd_b.assign(0)
            self.cd_d.assign(0)

            # Positive phase
            cm.dot(self.W,gpu_data,self.gpu_h)
            self.gpu_h.add_col_vec(self.c)
            cm.dot(self.gpu_target_vectors,gpu_target_for_data,self.gpu_target_vec_pos)
            self.gpu_h.add_dot(self.U,self.gpu_target_vec_pos)
            self.gpu_h.apply_sigmoid()
            
            if n_first_update != None:
                # A simple fix for having a non-zero gradient only for n_first_update examples
                self.gpu_target_vec_pos.slice(n_first_update,self.minibatch_size).assign(0)
                self.gpu_h.slice(n_first_update,self.minibatch_size).assign(0)

            self.cd_W.subtract_dot(self.gpu_h,gpu_data.T)
            self.cd_U.subtract_dot(self.gpu_h,self.gpu_target_vec_pos.T)
            self.cd_c.add_sums(self.gpu_h,axis=1,mult=-1.)
            self.cd_b.add_sums(gpu_data,axis=1,mult=-1.)
            self.cd_d.add_sums(self.gpu_target_vec_pos,axis=1,mult=-1.)

            if self.use_persistent_chain:
                cm.dot(self.W,self.gpu_x_persistent,self.gpu_h)
                self.gpu_h.add_col_vec(self.c)
                self.gpu_h.add_dot(self.U,self.gpu_y_persistent)
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

                cm.dot(self.U.T,self.gpu_h_sample,self.gpu_y)
                self.gpu_y.add_col_vec(self.d)
                cm.dot(self.gpu_target_vectors.T,self.gpu_y,self.gpu_target_vec_neg)
                self.gpu_target_vec_neg.transpose(self.gpu_y_trans)
                self.gpu_y_trans.sum(axis=1,target=self.gpu_y_trans_mean)
                self.gpu_y_trans_mean.divide(-self.n_classes)
                self.gpu_y_trans.add_col_vec(self.gpu_y_trans_mean)
                cm.exp(self.gpu_y_trans,target=self.gpu_y_trans)
                self.gpu_y_trans.sum(axis=1,target=self.gpu_y_trans_norm)
                for c in range(self.n_classes):                
                    self.gpu_y_trans.slice(c,c+1).divide(self.gpu_y_trans_norm)
                self.gpu_y_trans.transpose(self.gpu_y)

                # Up pass
                cm.dot(self.W,self.gpu_x_sample,self.gpu_h)
                self.gpu_h.add_col_vec(self.c)
                cm.dot(self.gpu_target_vectors,self.gpu_y,self.gpu_target_vec_neg)
                self.gpu_h.add_dot(self.U,self.gpu_target_vec_neg)
                self.gpu_h.apply_sigmoid()
            
            if self.use_persistent_chain:
                # Remember Gibbs chain's state
                self.gpu_x_persistent.assign(self.gpu_x_sample)
                self.gpu_y_persistent.assign(self.gpu_target_vec_neg)

            if n_first_update != None:
                self.gpu_x_sample.slice(n_first_update,self.minibatch_size).assign(0)
                self.gpu_target_vec_neg.slice(n_first_update,self.minibatch_size).assign(0)
                self.gpu_h.slice(n_first_update,self.minibatch_size).assign(0)

            
            self.cd_W.add_dot(self.gpu_h,self.gpu_x_sample.T)
            self.cd_U.add_dot(self.gpu_h,self.gpu_target_vec_neg.T)
            self.cd_c.add_sums(self.gpu_h,axis=1)
            self.cd_b.add_sums(self.gpu_x_sample,axis=1)
            self.cd_d.add_sums(self.gpu_target_vec_neg,axis=1)

            # Update RBM
            if is_labeled:
                alpha = self.gen_learning_weight
            else:
                alpha = self.semisup_learning_weight

            self.dW.add_mult(self.cd_W,alpha=alpha)
            self.dU.add_mult(self.cd_U,alpha=alpha)
            self.dc.add_mult(self.cd_c,alpha=alpha)
            self.db.add_mult(self.cd_b,alpha=alpha)
            self.dd.add_mult(self.cd_d,alpha=alpha)

        if n_first_update == None:
            lr = self.learning_rate/self.minibatch_size
        else:
            lr = self.learning_rate/n_first_update

        self.W.add_mult(self.dW,alpha=-lr)
        self.U.add_mult(self.dU,alpha=-lr)
        self.c.add_mult(self.dc,alpha=-lr)
        self.b.add_mult(self.db,alpha=-lr)
        self.d.add_mult(self.dd,alpha=-lr)

        #if self.print_first_row:
        #    gpu_data.copy_to_host()
        #    print gpu_data.numpy_array[:,0]
        #    self.gpu_x.copy_to_host()
        #    print self.gpu_x.numpy_array[:,0]

        ## Compute reconstruction error
        #self.gpu_x.subtract(gpu_data)
        #err = self.gpu_x.euclid_norm()
        #err = err**2
        #err /= self.gpu_x.shape[1]
        #return err

    def use(self,dataset):
        """
        Outputs the class prediction and distribution.
        """
        
        pred_class = np.zeros((len(dataset)))
        pred_probs = np.zeros((len(dataset),self.n_classes))
        n_loaded = 0
        self.gpu_x.copy_to_host()
        for t,example in enumerate(dataset):
            input,target = example
            # load some data on GPU
            self.gpu_x.numpy_array[:,n_loaded] = input
            n_loaded += 1
            if n_loaded >= self.minibatch_size:
                self.gpu_x.copy_to_device()
                self.compute_output(self.gpu_x)
                self.gpu_p_y_given_x.copy_to_host()
                pred_class[(t+1-self.minibatch_size):(t+1)] = self.gpu_p_y_given_x.numpy_array.argmax(axis=1)
                pred_probs[(t+1-self.minibatch_size):(t+1),:] = self.gpu_p_y_given_x.numpy_array
                n_loaded = 0
                
        # Compute for last examples!
        if n_loaded > 0:
            self.gpu_x.copy_to_device()
            self.compute_output(self.gpu_x)
            self.gpu_p_y_given_x.copy_to_host()
            pred_class[(len(dataset)-n_loaded):] = self.gpu_p_y_given_x.numpy_array.argmax(axis=1)[:n_loaded]
            pred_probs[(len(dataset)-n_loaded):,:] = self.gpu_p_y_given_x.numpy_array[:n_loaded,:]
            
        return zip(pred_class,pred_probs)

    def test(self,dataset):
        """
        Outputs the classifiaction error for each example.
        """
        outputs = self.use(dataset)
        costs = np.zeros(len(dataset),1)
        for o,c,example in zip(outputs,costs,dataset):
            x,y = example
            c[0] = int(y!=o[0])

        return outputs,costs
