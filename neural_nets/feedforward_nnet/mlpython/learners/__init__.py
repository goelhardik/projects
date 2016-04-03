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
MLPython supports many types of learning algorithms or "Learners".
A Learner object will always define the four following methods:

* ``train(self,trainset)``: runs the learning algorithm on ``trainset``.
* ``forget(self)``: resets the Learner to it's original state.
* ``use(self,dataset)``: computes and returns the output of the Learner for ``dataset``. 
  The method should return an iterator over these outputs.
* ``test(self,dataset)``: computes and returns the outputs of the Learner as well as the cost of 
  those outputs for ``dataset``. The method should return a pair of two iterators, the first
  being over the outputs and the second over the costs.

Of course, a constructor ``__init__(self,...)`` also needs to be defined, taking as argument
the different options or "hyper-parameters" this learning algorithm requires.

The ``learners`` package is divided into different modules or
subpackages, based on the task the associated Learners are trying to
solve of the type of data they require. 

The modules are:

* ``learners.generic``:          Learners not specific to a particular task or type of data.
* ``learners.classification``:   Learners for classification problems.
* ``learners.distribution``:          Learners for distribution or distrubtion estimation.
* ``learners.dynamic``:          Learners for sequential data.
* ``learners.features``:         Learners for feature extraction.

The subpackages are:

* ``learners.sparse``:           learners for data in sparse format.
* ``learners.third_party``:      learners based on third-party libraries.
* ``learners.gpu``:              learners running on GPUs.

"""
