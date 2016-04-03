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
The ``learners.third_party.libsvm.classification`` module contains 
an SVN classifier based on the LIBSVM library:

* SVMClassifier:  Classifier using LIBSVM's Support Vector Machine implementation.

"""


from mlpython.learners.generic import Learner
import numpy as np

try :
    import svm as libsvm
except ImportError:
    import warnings
    warnings.warn('\'import libsvm\' failed. The LIBSVM library is not properly installed. See mlpython/learners/third_party/libsvm/README for instructions.')


class SVMClassifier(Learner):
    """ 
    Classifier using LIBSVM's Support Vector Machine implementation
 
    Examples should be input and target pairs. The input can use
    either a sparse or coarse representation, as returned by the
    ``mlpython.misc.io.libsvm_load`` function.  Option ``kernel`` (which
    can be either ``'linear'``, ``'polynomial'``, ``'rbf'`` or
    ``'sigmoid'``) determines the type of kernel.

    Weights to examples of different classes can be given using 
    option ``label_weights``, which must be a dictionary mapping from 
    the label (string) to the weight (float).

    The SVM will also output probabilities if option ``'output_probabilities'``
    is True.

    Other options are the same as those in the LIBSVM implementation
    (see http://www.csie.ntu.edu.tw/~cjlin/libsvm for more details).

    **Required metadata:**

    * ``'targets'``
    * ``'class_to_id'``

    """
    def __init__(self,
                 kernel='linear',
                 degree=3,
                 gamma=1,
                 coef0=0,
                 C=1,
                 tolerance=0.001,
                 cache_size=100,
                 shrinking=True,
                 output_probabilities=False,
                 label_weights = None
                 ):
        
        self.kernel = kernel
        self.degree = degree
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.C = float(C)
        self.tolerance = float(tolerance)
        self.cache_size = cache_size
        self.shrinking = shrinking
        self.output_probabilities = output_probabilities
        self.label_weights = label_weights

    def train(self,trainset):
        """
        Trains the SVM.
        """

        self.n_classes = len(trainset.metadata['targets'])

        # Set LIBSVM parameters
        kernel_types = {'linear':libsvm.LINEAR,'polynomial':libsvm.POLY,
                        'rbf':libsvm.RBF,'sigmoid':libsvm.SIGMOID}
        if self.kernel not in kernel_types:
            raise ValueError('Invalid kernel: '+self.kernel+'. Should be either \'linear\', \'polynomial\', \'rbf\' or \'sigmoid\'')

        if self.label_weights != None:
            class_to_id = trainset.metadata['class_to_id']
            nr_weight = self.n_classes
            weight_label = range(self.n_classes)
            weight = [1]*self.n_classes
            for k,v in self.label_weights.iteritems():
                weight[class_to_id[k]] = v
        else:
            nr_weight = 0
            weight_label = []
            weight = []

        libsvm_params = libsvm.svm_parameter(svm_type = libsvm.C_SVC,
                                             kernel_type = kernel_types[self.kernel],
                                             degree=self.degree,
                                             gamma=self.gamma,
                                             coef0=self.coef0,
                                             C=self.C,
                                             probability=int(self.output_probabilities),
                                             cache_size=self.cache_size,
                                             eps=self.tolerance,
                                             shrinking=int(self.shrinking),
                                             nr_weight = nr_weight,
                                             weight_label = weight_label,
                                             weight = weight)
        

        # Put training set in the appropriate format:
        #  if is sparse (i.e. a pair), inputs are converted to dictionaries
        #  if not, inputs are assumed to be sequences and are kept intact
        libsvm_inputs = []
        libsvm_targets = []
        for input,target in trainset:
            if type(input) == tuple:
                libsvm_inputs += [dict(zip(input[1],input[0]))]
            else:
                libsvm_inputs += [input]
            libsvm_targets += [float(target)] # LIBSVM requires double-valued targets

        libsvm_problem = libsvm.svm_problem(libsvm_targets,libsvm_inputs)

        # Train SVM
        self.svm = libsvm.svm_model(libsvm_problem,libsvm_params)

    def forget(self):
        self.svm = None 
        self.n_classes = None

    def use(self,dataset):
        """
        Outputs the class_id chosen by the algorithm. If 
        ``output_probabilities`` is True, also outputs the vector
        of probabilities.
        """
        if self.output_probabilities:
            outputs = np.zeros((len(dataset),1+self.n_classes))
        else:
            outputs = np.zeros((len(dataset),1))
        for xy,out in zip(dataset,outputs):
            x,y = xy
            if type(x) == tuple:
                x = dict(zip(x[1],x[0]))

            if self.output_probabilities:
                c,probs = self.svm.predict_probability(x)
                out[0] = c
                for k,v in probs.iteritems():
                    out[k+1]=v
            else:
                c = self.svm.predict(x)
                out[0] = c
            
        return outputs

    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
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
