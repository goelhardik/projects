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

from generic import Learner
import numpy as np
from numpy import dot,ones,zeros,log,argmax,sum,sqrt,roots,array,diag,eye,pi,inf,nan,isnan,iscomplex,reshape,multiply,divide,add,absolute,cos,abs,arccos,sign,exp,minimum,maximum 
#from numpy.linalg import inv, eigvals
from numpy.random.mtrand import RandomState
import scipy.linalg
from scipy.special import gammaln
from mlpython.mathutils.linalg import product_matrix_vector, product_matrix_matrix, outer, solve, lu, getdiag, setdiag
#from mathutils.linalg_debug import product_matrix_vector, product_matrix_matrix, outer, solve, lu, getdiag, setdiag
import sys, time

# The training set for these models should be an iterator over matrices, where
# each matrix is a sequence and each row is an observation (vector)
# from that sequence.

class LinearDynamicalSystem(Learner):
    """ 
    Linear Dynamical System (LDS)
 
    This is a standard linear dynamical system, trained by EM.

    The options are

    * ``n_stages``
    * ``latent_size``
    * ``latent_covariance_matrix_regularizer``
    * ``input_covariance_matrix_regularizer``
    * ``latent_transition_matrix_regularizer``
    * ``input_transition_matrix_regularizer``
    * ``seed'

    **Required metadata:**

    * ``'input_size'``

    | **Reference:** 
    | Pattern Recognition and Machine Learning
    | Christopher M. Bishop
    | http://research.microsoft.com/en-us/um/people/cmbishop/prml/
    | Note: I tried to use the same notation. The only exception is that
    | I refer to the latent covariance matrix \Gamma as E here.
    """
    def __init__(   self,
                    n_stages= sys.maxint, # Maximum number of iterations on the training set
                    latent_size = 10, # Size of the latent variable
                    latent_covariance_matrix_regularizer = 0,
                    input_covariance_matrix_regularizer = 0,
                    latent_transition_matrix_regularizer = 0,
                    input_transition_matrix_regularizer = 0,
                    observation_variance = -1.,
                    seed = 1827,
                    ):
        self.stage = 0
        self.n_stages = n_stages
        self.latent_size = latent_size
        self.seed = seed
        self.rng = RandomState(seed)
        self.latent_covariance_matrix_regularizer = float(latent_covariance_matrix_regularizer)
        self.input_covariance_matrix_regularizer = float(input_covariance_matrix_regularizer)
        self.latent_transition_matrix_regularizer = float(latent_transition_matrix_regularizer)
        self.input_transition_matrix_regularizer = float(input_transition_matrix_regularizer) 
        self.observation_variance = float(observation_variance)

    def multivariate_norm_log_pdf(self,x,mu,cov):
        # -0.5 * (dot(x-mu,dot(inv(cov),x-mu)) + len(x)*log(2*pi) + log(det(cov)))
        self.vec_d_y[:] = x
        self.vec_d_y[:] -= mu
        solve(cov,reshape(self.vec_d_y,(-1,1)),reshape(self.vec_d_y2,(-1,1)),
              self.covf,self.colvecf,self.pivotscov)
        ret = dot(self.vec_d_y,self.vec_d_y2)
        ret += len(x)*log(2*pi)
        lu(cov,self.pcov,self.Lcov,self.Ucov,self.covf,self.pivotscov)
        getdiag(self.Ucov,self.vec_d_y)
        absolute(self.vec_d_y,self.vec_d_y2)
        log(self.vec_d_y2,self.vec_d_y)
        ret += sum(self.vec_d_y)
        ret *= -0.5
        return ret

    def EM_step(self,y_set,return_mu_post = False):
        """
        Computes the posterior statistics and outputs the M step
        estimates of the parameters.
        The set of probabilities p(y_t | y_{t-1}, ... , y_1) are also given.
        """

        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        mu_zero = self.mu_zero
        V_zero = self.V_zero
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E

        # Variables for estimating new parameters
        A_new = zeros((d_z,d_z))
        C_new = zeros((d_y,d_z))
        E_new = zeros((d_z,d_z))
        Sigma_new = zeros((d_y,d_y))
        mu_zero_new = zeros((d_z))
        V_zero_new = zeros((d_z,d_z))
        
        z_n_z_n_1_post_sum = zeros((d_z,d_z))
        z_n_z_n_post_sum = zeros((d_z,d_z))
        z_n_z_n_post_sum_no_last = zeros((d_z,d_z))
        z_n_z_n_post_sum_no_first = zeros((d_z,d_z))
        z_n_z_n_post_sum_first = zeros((d_z,d_z))
        outer_z_n_z_n_post_sum_first = zeros((d_z,d_z))
        z_n_post_sum_first = zeros((d_z))
        y_n_z_n_post_sum = zeros((d_y,d_z))
        y_n_y_n_sum = zeros((d_y,d_y))

        cond_probs = []

        # Temporary variable, to avoid memory allocation
        vec_d_z = zeros(d_z)
        vec_d_y = zeros(d_y)
        mat_d_z_d_z = zeros((d_z,d_z))
        mat_d_z_d_z2 = zeros((d_z,d_z))
        eye_d_z = eye(d_z)
        mat_times_C_trans = zeros((d_z,d_y))
        pred = zeros(d_y)
        cov_pred = zeros((d_y,d_y))
        K = zeros((d_z,d_y))
        KC = zeros((d_z,d_z))
        J = zeros((d_z,d_z))
        A_times_prev_mu = zeros(d_z)
        Af_d_y_d_y = zeros((d_y,d_y),order='fortran') # Temporary variables
        Bf_d_y_d_z = zeros((d_y,d_z),order='fortran') # for calls to
        Af_d_z_d_z = zeros((d_z,d_z),order='fortran') # math.linalg.solve(...)
        Bf_d_z_d_z = zeros((d_z,d_z),order='fortran') 
        pivots_d_y = zeros((d_y),dtype='i',order='fortran') 
        pivots_d_z = zeros((d_z),dtype='i',order='fortran') 
        z_n_z_n_1_post = zeros((d_z,d_z))
        z_n_z_n_post = zeros((d_z,d_z))
        y_n_z_n_post = zeros((d_y,d_z))
        y_n_y_n = zeros((d_y,d_y))
        T_sum = 0

        if return_mu_post:
            mu_post = []

        for y_t in y_set:
            T = len(y_t)
            T_sum += T
            mu_kalman_t = zeros((T,d_z))     # Filtering mus
            E_kalman_t = zeros((T,d_z,d_z))  # Filtering Es
            mu_post_t = zeros((T,d_z))       # Posterior mus (could be removed and computed once)
            E_post_t = zeros((T,d_z,d_z))    # Posterior Es  (could be removed and computed once)
            P_t = zeros((T-1,d_z,d_z)) 
            cond_probs_t = zeros(T)

            # Forward pass

            # Initialization at n = 0
            A_times_prev_mu[:] = 0
            product_matrix_matrix(V_zero,C.T,mat_times_C_trans)
            product_matrix_vector(C,mu_zero,pred)
            product_matrix_matrix(C,mat_times_C_trans,cov_pred)
            cov_pred += Sigma
            solve(cov_pred,mat_times_C_trans.T,K.T,Af_d_y_d_y,Bf_d_y_d_z,pivots_d_y)
            
            vec_d_y[:] = y_t[0]
            vec_d_y -= pred
            product_matrix_vector(K,vec_d_y,mu_kalman_t[0])
            mu_kalman_t[0] += mu_zero

            product_matrix_matrix(K,C,KC)
            mat_d_z_d_z[:] = eye_d_z
            mat_d_z_d_z -= KC
            product_matrix_matrix(mat_d_z_d_z,V_zero,E_kalman_t[0])
            cond_probs_t[0] = self.multivariate_norm_log_pdf(y_t[0],pred,cov_pred)
            # from n=1 to T-1
            for n in xrange(T-1):
                P_tn = P_t[n]
                product_matrix_matrix(E_kalman_t[n],A.T,mat_d_z_d_z)
                product_matrix_matrix(A,mat_d_z_d_z,P_tn)
                P_tn += E
                product_matrix_vector(A,mu_kalman_t[n],A_times_prev_mu)
                product_matrix_matrix(P_tn,C.T,mat_times_C_trans)
                product_matrix_vector(C,A_times_prev_mu,pred)
                product_matrix_matrix(C,mat_times_C_trans,cov_pred)
                cov_pred += Sigma
                solve(cov_pred,mat_times_C_trans.T,K.T,Af_d_y_d_y,Bf_d_y_d_z,pivots_d_y)
                vec_d_y[:] = y_t[n+1]
                vec_d_y -= pred
                product_matrix_vector(K,vec_d_y,mu_kalman_t[n+1])
                mu_kalman_t[n+1] += A_times_prev_mu
                
                product_matrix_matrix(K,C,KC)
                mat_d_z_d_z[:] = eye_d_z
                mat_d_z_d_z -= KC
                product_matrix_matrix(mat_d_z_d_z,P_tn,mat_d_z_d_z2)
                # To ensure symmetry
                E_kalman_t[n+1] = mat_d_z_d_z2
                E_kalman_t[n+1] += mat_d_z_d_z2.T
                E_kalman_t[n+1] /= 2
                cond_probs_t[n+1] = self.multivariate_norm_log_pdf(y_t[n+1],pred,cov_pred)


            mu_post_t[-1] = mu_kalman_t[-1]
            E_post_t[-1] = E_kalman_t[-1]

            # Compute last step statistics
            outer(mu_post_t[-1],mu_post_t[-1],z_n_z_n_post)
            z_n_z_n_post += E_post_t[-1]
            outer(y_t[-1],mu_post_t[-1],y_n_z_n_post)
            outer(y_t[-1],y_t[-1],y_n_y_n)
            # Update cumulative statistics
            z_n_z_n_post_sum += z_n_z_n_post
            z_n_z_n_post_sum_no_first += z_n_z_n_post
            y_n_z_n_post_sum += y_n_z_n_post
            y_n_y_n_sum += y_n_y_n

            # Backward pass
            pred[:] = 0
            cov_pred[:] = 0
            for n in xrange(T-2,-1,-1):
                P_tn = P_t[n]
                solve(P_tn.T,A,mat_d_z_d_z,Af_d_z_d_z,Bf_d_z_d_z,pivots_d_z)
                product_matrix_matrix(E_kalman_t[n],mat_d_z_d_z.T,J)
                product_matrix_vector(A,mu_kalman_t[n],vec_d_z)

                vec_d_z *= -1
                vec_d_z += mu_post_t[n+1]
                product_matrix_vector(J,vec_d_z,mu_post_t[n])
                mu_post_t[n] += mu_kalman_t[n]

                mat_d_z_d_z[:] = E_post_t[n+1]
                mat_d_z_d_z -= P_tn
                product_matrix_matrix(mat_d_z_d_z,J.T,mat_d_z_d_z2)
                product_matrix_matrix(J,mat_d_z_d_z2,mat_d_z_d_z)
                # To ensure symmetry
                E_post_t[n] = E_kalman_t[n]
                E_post_t[n] += mat_d_z_d_z
                E_post_t[n] += E_kalman_t[n].T
                E_post_t[n] += mat_d_z_d_z.T
                E_post_t[n] /= 2

                # Compute posterior statistics
                product_matrix_matrix(J,E_post_t[n+1],z_n_z_n_1_post)
                outer(mu_post_t[n+1],mu_post_t[n],mat_d_z_d_z)
                z_n_z_n_1_post += mat_d_z_d_z

                outer(mu_post_t[n],mu_post_t[n],z_n_z_n_post)
                z_n_z_n_post += E_post_t[n]
                
                outer(y_t[n],mu_post_t[n],y_n_z_n_post)
                outer(y_t[n],y_t[n],y_n_y_n)
                 
                # Update cumulative statistics
                z_n_z_n_1_post_sum += z_n_z_n_1_post
                z_n_z_n_post_sum += z_n_z_n_post
                if n > 0: 
                    z_n_z_n_post_sum_no_first += z_n_z_n_post
                else: 
                    z_n_z_n_post_sum_first += z_n_z_n_post
                    z_n_post_sum_first += mu_post_t[n]
                    outer(mu_post_t[n],mu_post_t[n],mat_d_z_d_z)
                    outer_z_n_z_n_post_sum_first += mat_d_z_d_z
                z_n_z_n_post_sum_no_last += z_n_z_n_post
                y_n_z_n_post_sum += y_n_z_n_post
                y_n_y_n_sum += y_n_y_n
            
            cond_probs += [cond_probs_t]

            if return_mu_post:
                mu_post += [mu_post_t]
        
        # Compute the M step estimates of the parameters
        #A_new = dot(z_n_z_n_1_post_sum,inv(z_n_z_n_post_sum_no_last+
        #                               eye_d_z*self.latent_transition_matrix_regularizer))
        solve(z_n_z_n_post_sum_no_last+eye_d_z*self.latent_transition_matrix_regularizer,
              z_n_z_n_1_post_sum.T,A_new.T)
        #C_new = dot(y_n_z_n_post_sum, inv(z_n_z_n_post_sum+
        #                                  eye_d_z*self.input_transition_matrix_regularizer))
        solve(z_n_z_n_post_sum+eye_d_z*self.input_transition_matrix_regularizer,
              y_n_z_n_post_sum.T,C_new.T)

        E_new[:] = z_n_z_n_post_sum_no_first
        z_n_z_n_1_A_T = dot(z_n_z_n_1_post_sum,A_new.T)
        E_new -= z_n_z_n_1_A_T.T
        E_new -= z_n_z_n_1_A_T # There is an error in Bishop's equation: the transpose on A is missing
        E_new += dot(A_new,dot(z_n_z_n_post_sum_no_last,A_new.T))
        E_new += eye_d_z*self.latent_covariance_matrix_regularizer
        E_new /= T_sum - len(y_set)
        Sigma_new[:] = y_n_y_n_sum
        C_z_n_y_n = dot(C_new,y_n_z_n_post_sum.T)
        Sigma_new -= C_z_n_y_n
        Sigma_new -= C_z_n_y_n.T # There is an error in Bishop's equation: the transpose on C is missing
        Sigma_new += dot(C_new,dot(z_n_z_n_post_sum,C_new.T)) # ... idem
        Sigma_new += eye(d_y)*self.input_covariance_matrix_regularizer
        Sigma_new /= T_sum

        mu_zero_new[:] = z_n_post_sum_first
        mu_zero_new /= len(y_set)
        V_zero_new[:] = z_n_z_n_post_sum_first
        V_zero_new -= outer_z_n_z_n_post_sum_first
        V_zero_new /= len(y_set)

        if return_mu_post:
            return (A_new,C_new,E_new,Sigma_new,mu_zero_new,V_zero_new),cond_probs,mu_post
        else:
            return (A_new,C_new,E_new,Sigma_new,mu_zero_new,V_zero_new),cond_probs

    def train(self,trainset):
        """
        Trains model with the EM algorithm, for (n_stages - stage) iterations. 
        If self.stage == 0, first initialize the model.
        """

        self.input_size = trainset.metadata['input_size']

        # Initialize model
        if self.stage == 0:
            self.forget()

        # Training with the EM algorithm
        #import time
        #print time.ctime()
        for it in xrange(self.stage,self.n_stages):
            params,cond_probs = self.EM_step(trainset)
            if self.observation_variance >= 0:
               self.A,self.C,self.E,dummy,self.mu_zero,self.V_zero = params
            else:
               self.A,self.C,self.E,self.Sigma,self.mu_zero,self.V_zero = params
            
            total_len = reduce(lambda x,y: x+len(y),cond_probs,0)
            print "NLL: ", -reduce(lambda x,y: x+sum(y),cond_probs,0)/total_len
            sys.stdout.flush()
            self.stage += 1
            #print time.ctime()
        del cond_probs

    def forget(self):
        d_y = self.input_size
        d_z = self.latent_size
        self.rng = RandomState(self.seed)
	rng = self.rng

        self.stage = 0 # Model will be untrained after initialization
        self.mu_zero = rng.randn(d_z)/d_z
        self.V_zero = diag(ones(d_z))
        self.A = rng.randn(d_z,d_z)/d_z
        self.C = rng.randn(d_y,d_z)/d_z
        if self.observation_variance >= 0:
           self.Sigma = diag(self.observation_variance*ones(d_y))
        else:
           self.Sigma = diag(ones(d_y))
        self.E = diag(ones(d_z))

        # Some useful temporary computation variables (mainly for multivariate_norm_log_pdf())
        self.vec_d_y = zeros((d_y))
        self.vec_d_y2 = zeros((d_y))
        self.pcov = zeros((d_y),dtype='i')
        self.Lcov = zeros((d_y,d_y))
        self.Ucov = zeros((d_y,d_y))
        self.covf = zeros((d_y,d_y),order='fortran')
        self.colvecf = zeros((d_y,1),order='fortran')
        self.pivotscov = zeros((d_y),dtype='i')

    def use(self,dataset):
        """
        Outputs the log-likelihood of the sequences in dataset
        """
        dummy, cond_probs = self.EM_step(dataset)
        outputs = array(map(sum,cond_probs))
        return reshape(outputs,(-1,1))

    def test(self,dataset):
        """
        Outputs the log-likelihood and average NLL (normalized by the length of
        each sequence) of the sequences in dataset
        """
        outputs = self.use(dataset)
        costs = zeros((len(outputs),1))
        # Compute normalized NLLs
        for seq,t in zip(dataset,xrange(len(dataset))):
            costs[t,0] = -outputs[t,0]/len(seq)

        return outputs,costs

class SparseLinearDynamicalSystem(Learner):
    """ 
    Sparse Linear Dynamical System (SLDS)
 
    This is a linear dynamical system where the latent space representation
    is encouraged to be sparse.

    Options:
    - 'n_stages'
    - 'latent_size'
    - 'latent_transition_matrix_regularizer'
    - 'emission_matrix_regularizer'
    - 'gamma_prior_alpha'
    - 'gamma_prior_beta'
    - 'max_Esteps'
    - 'gamma_change_tolerance'
    - 'output_laplace_probs'
    - 'seed'

    Required metadata:
    - 'input_size'

    """
    def __init__(   self,
                    n_stages= sys.maxint, # Maximum number of iterations on the training set
                    latent_size = 10, # Size of the latent variable
                    observation_variance = 0.001, # input_covariance_matrix_regularizer = 0.,
                    latent_transition_matrix_regularizer = 0.,
                    emission_matrix_regularizer = 0.,
                    gamma_prior_alpha = 1.,
                    gamma_prior_beta = 0.0001,
                    gamma_change_tolerance = 0.0001,
                    output_laplace_probs = True,
                    max_Esteps = 1,
                    max_test_Esteps = 25,
                    seed = 1827,
                    verbose = False
                    ):
        self.stage = 0
        self.n_stages = n_stages
        self.latent_size = latent_size
        self.seed = seed
        self.max_Esteps = max_Esteps
        self.max_test_Esteps = max_test_Esteps
        self.last_Esteps = max_Esteps
        self.rng = RandomState(seed)
        self.observation_variance = float(observation_variance)
        self.latent_transition_matrix_regularizer = float(latent_transition_matrix_regularizer)
        self.emission_matrix_regularizer = float(emission_matrix_regularizer)
        self.gamma_prior_alpha = float(gamma_prior_alpha)
        self.gamma_prior_beta = float(gamma_prior_beta)
        self.gamma_change_tolerance = float(gamma_change_tolerance)
        self.output_laplace_probs = output_laplace_probs
        self.verbose = verbose

    def multivariate_norm_log_pdf(self,x,mu,cov):
        #return -0.5 * (dot(x-mu,dot(inv(cov),x-mu)) + len(x)*log(2*pi) + sum(log(eigvals(cov))))
        #return_old =  -0.5 * (dot(x-mu,solve(cov,x-mu)) + len(x)*log(2*pi) + sum(log(abs(diag(scipy.linalg.lu(cov)[2])))))
        self.vec_d_y[:] = x
        self.vec_d_y[:] -= mu
        solve(cov,reshape(self.vec_d_y,(-1,1)),reshape(self.vec_d_y2,(-1,1)),
              self.covf,self.colvecf,self.pivotscov)
        ret = dot(self.vec_d_y,self.vec_d_y2)
        ret += len(x)*log(2*pi)
        lu(cov,self.pcov,self.Lcov,self.Ucov,self.covf,self.pivotscov)
        getdiag(self.Ucov,self.vec_d_y)
        absolute(self.vec_d_y,self.vec_d_y2)
        log(self.vec_d_y2,self.vec_d_y)
        ret += sum(self.vec_d_y)
        ret *= -0.5
        return ret

    def log_prior_gamma(self,gamma_tn):
        alpha = self.gamma_prior_alpha
        beta = self.gamma_prior_beta
        result = sum(alpha*log(beta)-gammaln(alpha)-(alpha+1)*log(gamma_tn)-beta/gamma_tn)
        return result

    def log_prior_log_gamma(self,gamma_tn):
        alpha = self.gamma_prior_alpha
        beta = self.gamma_prior_beta
        return sum(alpha*log(beta)-gammaln(alpha)-(alpha)*log(gamma_tn)-beta/gamma_tn)

    def log_det_diff2_log_gamma(self,A,E,zz_tn_prev,zz_tn,gamma_tn):
        d_z = len(gamma_tn)
        product_matrix_matrix(zz_tn_prev,A.T,self.mat_d_z_d_z)
        product_matrix_matrix(A,self.mat_d_z_d_z,self.mat_d_z_d_z2)
        getdiag(self.mat_d_z_d_z2,self.AzzA_prev)
        G = 0.5*diag(zz_tn)+self.gamma_prior_beta
        H = 0.5*self.AzzA_prev
        gamma_E_1 = (gamma_tn+E)
        gamma_E_2 = gamma_E_1*gamma_E_1
        gamma_E_3 = gamma_E_2*gamma_E_1
        return sum(log(G/gamma_tn+H*gamma_tn*(E-gamma_tn)/gamma_E_3-0.5*E*gamma_tn/gamma_E_2))

    def EM_step(self,y_set,gamma_set,training = False, return_mu_post = False):
        """
        Computes the posterior statistics and outputs the M step
        estimates of the parameters.
        Also outputs the non-parametric, sparsity inducing variances gamma_t.
        Optionally, can output the posterior means of the latent state variables.
        """
        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        #V_zero = self.V_zero
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E

        # Variables for estimating new parameters
        A_new = zeros((d_z,d_z))
        C_new = zeros((d_y,d_z))
        
        z_n_z_n_1_post_sum = zeros((d_z,d_z))
        z_n_z_n_post_sum = zeros((d_z,d_z))
        A_new_denums = zeros((d_z,d_z,d_z))
        y_n_z_n_post_sum = zeros((d_y,d_z))

        # Temporary variable, to avoid memory allocation
        vec_d_z = zeros(d_z)
        vec_d_z2 = zeros(d_z)
        vec_d_y = zeros(d_y)
        mat_d_z_d_z = zeros((d_z,d_z))
        mat_d_z_d_z2 = zeros((d_z,d_z))
        eye_d_z = eye(d_z)
        mat_times_C_trans = zeros((d_z,d_y))
        pred = zeros(d_y)
        cov_pred = zeros((d_y,d_y))
        A_gamma = zeros((d_z,d_z))
        E_gamma = zeros((d_z,d_z))
        K = zeros((d_z,d_y))
        KC = zeros((d_z,d_z))
        J = zeros((d_z,d_z))
        A_times_prev_mu = zeros(d_z)
        Af_d_y_d_y = zeros((d_y,d_y),order='fortran') # Temporary variables
        Bf_d_y_d_z = zeros((d_y,d_z),order='fortran') # for calls to
        Af_d_z_d_z = zeros((d_z,d_z),order='fortran') # math.linalg.solve(...)
        Bf_d_z_d_z = zeros((d_z,d_z),order='fortran') 
        pivots_d_y = zeros((d_y),dtype='i',order='fortran') 
        pivots_d_z = zeros((d_z),dtype='i',order='fortran') 
        z_n_z_n_1_post = zeros((d_z,d_z))
        z_n_z_n_post = zeros((d_z,d_z))
        weighted_z_n_z_n_post = zeros((d_z,d_z,d_z))
        next_z_n_z_n_post = zeros((d_z,d_z))
        y_n_z_n_post = zeros((d_y,d_z))

        if training == True:
            max_Esteps = self.max_Esteps
            last_Esteps = self.last_Esteps
        else:
            max_Esteps = self.max_test_Esteps
            last_Esteps = self.max_test_Esteps

        Esteps = 0
        have_A_denum = False
        get_A_denum = False
        finished = False
        while not finished:
            T_sum = 0
            gamma_mean_diff = 0
            z_n_z_n_1_post_sum[:] = 0
            z_n_z_n_post_sum[:] = 0
            y_n_z_n_post_sum[:] = 0
            A_new_denums[:] = 0

            Esteps += 1
            if Esteps == max_Esteps:
                get_A_denum = True
                finished = True
            elif Esteps >= last_Esteps:
                get_A_denum = True
                
            if return_mu_post:
                mu_post = []

            for y_t,gamma_t in zip(y_set,gamma_set):
                T = len(y_t)
                T_sum += T
                mu_kalman_t = zeros((T,d_z))     # Filtering mus
                E_kalman_t = zeros((T,d_z,d_z))  # Filtering Es
                mu_post_t = zeros((T,d_z))
                E_post_t = zeros((T,d_z,d_z))
                P_t = zeros((T-1,d_z,d_z)) 
            
                # Forward pass
            
                # Initialization at n = 0
                A_times_prev_mu[:] = 0
                multiply(C.T,reshape(gamma_t[0],(-1,1)),mat_times_C_trans)
                pred[:] = 0

                product_matrix_matrix(C,mat_times_C_trans,cov_pred)
                cov_pred += Sigma
                solve(cov_pred,mat_times_C_trans.T,K.T,Af_d_y_d_y,Bf_d_y_d_z,pivots_d_y)
            
                vec_d_y[:] = y_t[0]
                vec_d_y -= pred
                product_matrix_vector(K,vec_d_y,mu_kalman_t[0])

                product_matrix_matrix(K,C,KC)
                mat_d_z_d_z[:] = eye_d_z
                mat_d_z_d_z -= KC
                multiply(mat_d_z_d_z,gamma_t[0],E_kalman_t[0])

                # from n=1 to T-1
                for n in xrange(T-1):
                    divide(1.,E,vec_d_z)
                    divide(1.,gamma_t[n+1],vec_d_z2)
                    vec_d_z += vec_d_z2
                    divide(1.,vec_d_z,vec_d_z2)
                    setdiag(E_gamma,vec_d_z2) 
                    divide(E,gamma_t[n+1],vec_d_z)
                    vec_d_z += 1
                    divide(A,reshape(vec_d_z,(-1,1)),A_gamma)
                    P_tn = P_t[n]
                    product_matrix_matrix(E_kalman_t[n],A_gamma.T,mat_d_z_d_z)

                    product_matrix_matrix(A_gamma,mat_d_z_d_z,P_tn)
                    P_tn += E_gamma
                    product_matrix_vector(A_gamma,mu_kalman_t[n],A_times_prev_mu)
                    product_matrix_matrix(P_tn,C.T,mat_times_C_trans)
                    product_matrix_vector(C,A_times_prev_mu,pred)
                    product_matrix_matrix(C,mat_times_C_trans,cov_pred)
                    cov_pred += Sigma
                    solve(cov_pred,mat_times_C_trans.T,K.T,Af_d_y_d_y,Bf_d_y_d_z,pivots_d_y)
                    vec_d_y[:] = y_t[n+1]
                    vec_d_y -= pred
                    product_matrix_vector(K,vec_d_y,mu_kalman_t[n+1])
                    mu_kalman_t[n+1] += A_times_prev_mu
                    
                    product_matrix_matrix(K,C,KC)
                    mat_d_z_d_z[:] = eye_d_z
                    mat_d_z_d_z -= KC
                    product_matrix_matrix(mat_d_z_d_z,P_tn,mat_d_z_d_z2)
                    # To ensure symmetry
                    E_kalman_t[n+1] = mat_d_z_d_z2
                    E_kalman_t[n+1] += mat_d_z_d_z2.T
                    E_kalman_t[n+1] /= 2

                mu_post_t[-1] = mu_kalman_t[-1]
                E_post_t[-1] = E_kalman_t[-1]

                # Compute last step statistics
                outer(mu_post_t[-1],mu_post_t[-1],z_n_z_n_post)
                z_n_z_n_post += E_post_t[-1]
                outer(y_t[-1],mu_post_t[-1],y_n_z_n_post)
                # Update cumulative statistics
                z_n_z_n_post_sum += z_n_z_n_post
                y_n_z_n_post_sum += y_n_z_n_post
 
                # Backward pass
                pred[:] = 0
                cov_pred[:] = 0
                for n in xrange(T-2,-1,-1):
                    next_z_n_z_n_post[:] = z_n_z_n_post
                    divide(E,gamma_t[n+1],vec_d_z)
                    vec_d_z += 1
                    divide(A,reshape(vec_d_z,(-1,1)),A_gamma)

                    P_tn = P_t[n]
                    solve(P_tn.T,A_gamma,mat_d_z_d_z,Af_d_z_d_z,Bf_d_z_d_z,pivots_d_z)
                    product_matrix_matrix(E_kalman_t[n],mat_d_z_d_z.T,J)
                    product_matrix_vector(A_gamma,mu_kalman_t[n],vec_d_z)

                    vec_d_z *= -1
                    vec_d_z += mu_post_t[n+1]
                    product_matrix_vector(J,vec_d_z,mu_post_t[n])
                    mu_post_t[n] += mu_kalman_t[n]
    
                    mat_d_z_d_z[:] = E_post_t[n+1]
                    mat_d_z_d_z -= P_tn
                    product_matrix_matrix(mat_d_z_d_z,J.T,mat_d_z_d_z2)
                    product_matrix_matrix(J,mat_d_z_d_z2,mat_d_z_d_z)
                    # To ensure symmetry
                    E_post_t[n] = E_kalman_t[n]
                    E_post_t[n] += mat_d_z_d_z
                    E_post_t[n] += E_kalman_t[n].T
                    E_post_t[n] += mat_d_z_d_z.T
                    E_post_t[n] /= 2
    
                    # Compute posterior statistics
                    product_matrix_matrix(J,E_post_t[n+1],z_n_z_n_1_post)
                    outer(mu_post_t[n+1],mu_post_t[n],mat_d_z_d_z)
                    z_n_z_n_1_post += mat_d_z_d_z
    
                    outer(mu_post_t[n],mu_post_t[n],z_n_z_n_post)
                    z_n_z_n_post += E_post_t[n]
                    
                    outer(y_t[n],mu_post_t[n],y_n_z_n_post)
                     
                    # Update cumulative statistics
                    z_n_z_n_1_post_sum += z_n_z_n_1_post
                    z_n_z_n_post_sum += z_n_z_n_post
                    y_n_z_n_post_sum += y_n_z_n_post
                    
                    gamma_mean_diff += self.compute_gamma(A,E,z_n_z_n_post,next_z_n_z_n_post,gamma_t[n+1])
                    #print gamma_t[n+1]
                    if get_A_denum == True:
                        # Compute the denominator of the A update, 
                        # which requires d_z matrices of size (d_z,d_z)
                        # (i.e. d_z different weighted sums of the z_n_z_n_post matrices)
                        add(gamma_t[n+1],E,vec_d_z)
                        divide(gamma_t[n+1],vec_d_z,vec_d_z2)
                        multiply(reshape(z_n_z_n_post,(1,d_z,d_z)),reshape(vec_d_z2,(d_z,1,1)),weighted_z_n_z_n_post)
                        A_new_denums += weighted_z_n_z_n_post
                        have_A_denum = True
                        
                new_gamma = (diag(z_n_z_n_post)+2*self.gamma_prior_beta)/(2*self.gamma_prior_alpha+3)
                gamma_mean_diff += sum((gamma_t[0]-new_gamma)**2)/d_z
                
                gamma_t[0] = new_gamma
                
            gamma_mean_diff /= T_sum
            if gamma_mean_diff < self.gamma_change_tolerance:
                if training == True:
                    if have_A_denum == True:
                        finished = True
                        self.last_Esteps = Esteps
                    else:
                        get_A_denum = True
                else:
                    finished = True                    
            elif gamma_mean_diff <= 10*self.gamma_change_tolerance and training == True:
                get_A_denum = True
            if self.verbose:
                print gamma_mean_diff, max_Esteps, Esteps
            if return_mu_post:
                mu_post += [mu_post_t]

        # Compute the M step estimates of the parameters
        if training == True:
            for i in xrange(d_z):
                solve(A_new_denums[i]+eye(d_z)*self.latent_transition_matrix_regularizer,z_n_z_n_1_post_sum[i:(i+1)].T,A_new[i:(i+1)].T)
        
            solve(z_n_z_n_post_sum+eye_d_z*self.emission_matrix_regularizer,y_n_z_n_post_sum.T,C_new.T)
        
        if return_mu_post:
            return (A_new,C_new),gamma_set,mu_post
        else:
            return (A_new,C_new),gamma_set

    def compute_gamma(self,A,E,zz_tn_prev,zz_tn,gamma_tn):
        """
        Replaces the current value of the gamma parameters with 
        its updated value, and returns the mean square difference between the two.
        """
        # This is the main bottleneck of the code.
        # Would be faster if:
        # - implemented in C
        # - roots() was also implemented in C
        d_z = len(gamma_tn)
        product_matrix_matrix(zz_tn_prev,A.T,self.mat_d_z_d_z)
        product_matrix_matrix(A,self.mat_d_z_d_z,self.mat_d_z_d_z2)
        getdiag(self.mat_d_z_d_z2,self.AzzA_prev)
        G = diag(zz_tn)+2*self.gamma_prior_beta
        H = self.AzzA_prev
        a1 = 2.0*(self.gamma_prior_alpha+1.0)
        a2 = (4.0*self.gamma_prior_alpha+5.0)*E + H - G
        a3 = ((2.0*self.gamma_prior_alpha+3)*E-2.0*G)*E
        a4 = -G*E**2
        Q = ((3.0*a3/a1)-((a2/a1)**2))/9
        R = (9*a1*a2*a3-27*a4*(a1**2)-2*(a2**3))/(54*a1**3)
        ##delta = Q**3+R**2
        #rho = sqrt(-Q**3)
        #theta = arccos(R/rho)
        theta = arccos(sign(R)*minimum(exp(log(abs(R))-3.0/2.0*log(-Q)),1.0))
        #print theta1, theta
        #JJ = pow(rho,1.0/3)
        HH = sqrt(-Q)
        am = 2*HH*cos(theta/3)-a2/(3.0*a1)        
        am = maximum(abs(am),0.00001)
        
        gamma_mean_diff = sum((am-gamma_tn)**2)/d_z
        gamma_tn[:] = am
        return gamma_mean_diff

    def cond_probs(self,y_set,gamma_set):
        """
        Given the set of gamma variables, outputs the set of 
        probabilities p(y_t | y_{t-1}, ... , y_1, gamma_{t-1}, ... , gamma_1)
        """

        # Note (HUGO): this function should probably be implemented in C
        #              to make it much faster, since it requires for loops.

        # Setting variables with friendlier name
        d_y = self.input_size
        d_z = self.latent_size
        A = self.A
        C = self.C
        Sigma = self.Sigma
        E = self.E

        cond_probs = []
        map_probs = []
        laplace_probs = []
        y_pred = []
        
        z_n_z_n_post_sum = zeros((d_z,d_z))

        # Temporary variable, to avoid memory allocation
        vec_d_z = zeros(d_z)
        vec_d_z2 = zeros(d_z)
        vec_d_y = zeros(d_y)
        mat_d_z_d_z = zeros((d_z,d_z))
        mat_d_z_d_z2 = zeros((d_z,d_z))
        eye_d_z = eye(d_z)
        mat_times_C_trans = zeros((d_z,d_y))
        pred = zeros(d_y)
        cov_pred = zeros((d_y,d_y))
        A_gamma = zeros((d_z,d_z))
        E_gamma = zeros((d_z,d_z))
        K = zeros((d_z,d_y))
        KC = zeros((d_z,d_z))
        J = zeros((d_z,d_z))
        A_times_prev_mu = zeros(d_z)
        Af_d_y_d_y = zeros((d_y,d_y),order='fortran') # Temporary variables
        Bf_d_y_d_z = zeros((d_y,d_z),order='fortran') # for calls to
        Af_d_z_d_z = zeros((d_z,d_z),order='fortran') # math.linalg.solve(...)
        Bf_d_z_d_z = zeros((d_z,d_z),order='fortran') 
        pivots_d_y = zeros((d_y),dtype='i',order='fortran') 
        pivots_d_z = zeros((d_z),dtype='i',order='fortran') 
        z_n_z_n_post = zeros((d_z,d_z))
        next_z_n_z_n_post = zeros((d_z,d_z))

        log_det_diff2_log_gamma = 0
        for y_t,gamma_t in zip(y_set,gamma_set):
            T = len(y_t)
            cond_probs_t = zeros(T)
            map_probs_t = zeros(T)
            laplace_probs_t = zeros(T)
            y_pred_t = zeros((T,d_y))
            mu_kalman_t = zeros((T,d_z))     # Filtering mus
            E_kalman_t = zeros((T,d_z,d_z))  # Filtering Es
            mu_post_t = zeros((T,d_z))
            E_post_t = zeros((T,d_z,d_z))
            P_t = zeros((T-1,d_z,d_z)) 
            
            # Forward pass
            
            # Initialization at n = 0
            A_times_prev_mu[:] = 0
            multiply(C.T,reshape(gamma_t[0],(-1,1)),mat_times_C_trans)
            pred[:] = 0
            product_matrix_matrix(C,mat_times_C_trans,cov_pred)
            cov_pred += Sigma
            solve(cov_pred,mat_times_C_trans.T,K.T,Af_d_y_d_y,Bf_d_y_d_z,pivots_d_y)
            
            vec_d_y[:] = y_t[0]
            vec_d_y -= pred
            product_matrix_vector(K,vec_d_y,mu_kalman_t[0])
            
            product_matrix_matrix(K,C,KC)
            mat_d_z_d_z[:] = eye_d_z
            mat_d_z_d_z -= KC
            multiply(mat_d_z_d_z,gamma_t[0],E_kalman_t[0])

            cond_probs_t[0] = self.multivariate_norm_log_pdf(y_t[0],pred,cov_pred)
            y_pred_t[0] = pred
            # from n=1 to T-1
            for n in xrange(T-1):
                divide(1.,E,vec_d_z)
                divide(1.,gamma_t[n+1],vec_d_z2)
                vec_d_z += vec_d_z2
                divide(1.,vec_d_z,vec_d_z2)
                setdiag(E_gamma,vec_d_z2) 
                divide(E,gamma_t[n+1],vec_d_z)
                vec_d_z += 1
                divide(A,reshape(vec_d_z,(-1,1)),A_gamma)
                
                P_tn = P_t[n]
                product_matrix_matrix(E_kalman_t[n],A_gamma.T,mat_d_z_d_z)
                product_matrix_matrix(A_gamma,mat_d_z_d_z,P_tn)
                P_tn += E_gamma
                product_matrix_vector(A_gamma,mu_kalman_t[n],A_times_prev_mu)
                product_matrix_matrix(P_tn,C.T,mat_times_C_trans)
                product_matrix_vector(C,A_times_prev_mu,pred)
                product_matrix_matrix(C,mat_times_C_trans,cov_pred)
                cov_pred += Sigma
                solve(cov_pred,mat_times_C_trans.T,K.T,Af_d_y_d_y,Bf_d_y_d_z,pivots_d_y)
                vec_d_y[:] = y_t[n+1]
                vec_d_y -= pred
                product_matrix_vector(K,vec_d_y,mu_kalman_t[n+1])
                mu_kalman_t[n+1] += A_times_prev_mu
                
                product_matrix_matrix(K,C,KC)
                mat_d_z_d_z[:] = eye_d_z
                mat_d_z_d_z -= KC
                product_matrix_matrix(mat_d_z_d_z,P_tn,mat_d_z_d_z2)
                # To ensure symmetry
                E_kalman_t[n+1] = mat_d_z_d_z2
                E_kalman_t[n+1] += mat_d_z_d_z2.T
                E_kalman_t[n+1] /= 2

                mu_post_t[-1] = mu_kalman_t[-1]
                E_post_t[-1] = E_kalman_t[-1]

                # Compute last step statistics
                outer(mu_post_t[-1],mu_post_t[-1],z_n_z_n_post)
                z_n_z_n_post += E_post_t[-1]
                # Update cumulative statistics
                z_n_z_n_post_sum += z_n_z_n_post

                cond_probs_t[n+1] = self.multivariate_norm_log_pdf(y_t[n+1],pred,cov_pred)
                y_pred_t[n+1] = pred

                
            #print y_t, y_pred_t
            # Backward pass
            pred[:] = 0
            cov_pred[:] = 0
            for n in xrange(T-2,-1,-1):
                next_z_n_z_n_post[:] = z_n_z_n_post
                divide(E,gamma_t[n+1],vec_d_z)
                vec_d_z += 1
                divide(A,reshape(vec_d_z,(-1,1)),A_gamma)
                
                P_tn = P_t[n]
                solve(P_tn.T,A_gamma,mat_d_z_d_z,Af_d_z_d_z,Bf_d_z_d_z,pivots_d_z)
                product_matrix_matrix(E_kalman_t[n],mat_d_z_d_z.T,J)
                product_matrix_vector(A_gamma,mu_kalman_t[n],vec_d_z)
                
                vec_d_z *= -1
                vec_d_z += mu_post_t[n+1]
                product_matrix_vector(J,vec_d_z,mu_post_t[n])
                mu_post_t[n] += mu_kalman_t[n]
                
                mat_d_z_d_z[:] = E_post_t[n+1]
                mat_d_z_d_z -= P_tn
                product_matrix_matrix(mat_d_z_d_z,J.T,mat_d_z_d_z2)
                product_matrix_matrix(J,mat_d_z_d_z2,mat_d_z_d_z)
                # To ensure symmetry
                E_post_t[n] = E_kalman_t[n]
                E_post_t[n] += mat_d_z_d_z
                E_post_t[n] += E_kalman_t[n].T
                E_post_t[n] += mat_d_z_d_z.T
                E_post_t[n] /= 2
                    
                outer(mu_post_t[n],mu_post_t[n],z_n_z_n_post)
                z_n_z_n_post += E_post_t[n]

                dummy = self.compute_gamma(A,E,z_n_z_n_post,next_z_n_z_n_post,gamma_t[n+1])
                log_prior_gamma = self.log_prior_gamma(gamma_t[n+1])
                #print log_prior_gamma
                log_prior_log_gamma = self.log_prior_log_gamma(gamma_t[n+1])               
                log_det_diff2_log_gamma = self.log_det_diff2_log_gamma(A,E,z_n_z_n_post,next_z_n_z_n_post,gamma_t[n+1])
                map_probs_t[n+1] = cond_probs_t[n+1]+log_prior_gamma
                laplace_probs_t[n+1] = cond_probs_t[n+1]+log_prior_log_gamma+d_z*log(2*pi)/2-0.5*log_det_diff2_log_gamma

            gamma_t[0] = (diag(z_n_z_n_post)+2*self.gamma_prior_beta)/(2*self.gamma_prior_alpha+3)
            log_prior_gamma = self.log_prior_gamma(gamma_t[0])
            log_prior_log_gamma = self.log_prior_log_gamma(gamma_t[0])
            log_det_diff2_log_gamma = sum((z_n_z_n_post/2+self.gamma_prior_beta)/gamma_t[0])
            map_probs_t[0] = cond_probs_t[0]+log_prior_gamma
            laplace_probs_t[0] = cond_probs_t[0]+log_prior_log_gamma+d_z*log(2*pi)/2-0.5*log_det_diff2_log_gamma

            cond_probs += [cond_probs_t]
            map_probs += [map_probs_t]
            laplace_probs += [laplace_probs_t]
            y_pred += [y_pred_t]
        
        return cond_probs, map_probs, laplace_probs, y_pred


    def train(self,trainset):
        """
        Trains model with the EM algorithm, for (n_stages - stage) iterations. 
        If self.stage == 0, first initialize the model.
        """

        self.input_size = trainset.metadata['input_size']        

        # Initialize model
        if self.stage == 0:
            self.forget()

        # Initialize the gammas
        if self.stage == 0:
            self.gamma_set = []
            for seq in trainset:
                self.gamma_set = self.gamma_set + [ones((len(seq),self.latent_size))]

        # Training with the EM algorithm
        for it in xrange(self.stage,self.n_stages):
            # E step
            (self.A,self.C),new_gamma_set = self.EM_step(trainset,self.gamma_set,training = True)
            self.gamma_set = new_gamma_set
            cond_probs,map_probs,laplace_probs,y_pred = self.cond_probs(trainset,self.gamma_set)
            total_len = reduce(lambda x,y: x+len(y),cond_probs,0)
            print "Training Stage: ",it
            print "LogLike: ", reduce(lambda x,y: x+sum(y),cond_probs,0)/total_len
            print "LogMAP: ", reduce(lambda x,y: x+sum(y),map_probs,0)/total_len
            print "LogMarg: ", reduce(lambda x,y: x+sum(y),laplace_probs,0)/total_len
            sys.stdout.flush()
            self.stage += 1

    def forget(self):
        d_y = self.input_size
        d_z = self.latent_size
        self.rng = RandomState(self.seed)
	rng = self.rng

        self.stage = 0 # Model will be untrained after initialization
        self.A = rng.randn(d_z,d_z)/d_z
        self.C = rng.randn(d_y,d_z)/d_z
        self.Sigma = self.observation_variance*diag(ones(d_y))
        self.E = ones(d_z)

        # Some useful temporary computation variables (mainly for multivariate_norm_log_pdf()
        # and compute_gamma() )
        self.mat_d_z_d_z = zeros((d_z,d_z))
        self.mat_d_z_d_z2 = zeros((d_z,d_z))
        self.AzzA_prev = zeros((d_z))
        self.vec_d_y = zeros((d_y))
        self.vec_d_y2 = zeros((d_y))
        self.pcov = zeros((d_y),dtype='i')
        self.Lcov = zeros((d_y,d_y))
        self.Ucov = zeros((d_y,d_y))
        self.covf = zeros((d_y,d_y),order='fortran')
        self.colvecf = zeros((d_y,1),order='fortran')
        self.pivotscov = zeros((d_y),dtype='i')

    def use(self,dataset):
        """
        Outputs the log-likelihood of the sequences in dataset
        """
        # Initialize gamma_set
        gamma_set = []
        for seq in dataset:
            gamma_set = gamma_set + [ones((len(seq),self.latent_size))]

        dummy, new_gamma_set = self.EM_step(dataset,gamma_set)
        cond_probs,map_probs,laplace_probs,y_pred = self.cond_probs(dataset,gamma_set)
        if self.output_laplace_probs:
           outputs = laplace_probs
        else:
           outputs = map_probs

        outputs = array(map(sum,outputs))
        #import code; code.interact(local=locals())
        
        return reshape(outputs,(-1,1))

    def test(self,dataset):
        """
        Outputs the log-likelihood and average NLL (normalized by the length of
        each sequence) of the sequences in dataset
        """
        outputs = self.use(dataset)
        costs = zeros((len(outputs),1))
        # Compute normalized NLLs
        for seq,t in zip(dataset,xrange(len(dataset))):
            costs[t,0] = -outputs[t,0]/len(seq)

        return outputs,costs
        
