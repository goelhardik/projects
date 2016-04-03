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
The ``mathutils.nonlinear`` module contains several useful nonlinear
functions on NumPy arrays. All functions avoid memory allocation, by
requiring the NumPy array in which to write the answer. All arrays
should be double arrays.

This module defines the following functions:

* ``sigmoid``:         Computes the sigmoid function.
* ``dsigmoid``:        Computes the derivative of a sigmoid function with respect to its input.
* ``reclin``:          Computes the rectified linear function.
* ``dreclin``:         Computes the derivative of a rectified linear function with respect to its input.
* ``softplus``:        Computes the softplus function.
* ``softmax``:         Computes the softmax function.

"""

import numpy as np
import nonlinear_

def sigmoid(input,output):
    """
    Computes the sigmoid function sigm(input) = 1/(1+exp(-input)) = output
    """
    nonlinear_.sigmoid_(input,output)

def dsigmoid(output,doutput,dinput):
    """
    Computes the derivative of a sigmoid function with respect to its input, 
    given the output of the sigmoid and the derivative on the output.
    """
    nonlinear_.dsigmoid_(output,doutput,dinput)

def reclin(input,output):
    """
    Computes the rectified linear function reclin(input) = 1_{input>0}*input = output
    """
    nonlinear_.reclin_(input,output)

def dreclin(output,doutput,dinput):
    """
    Computes the derivative of a rectified linear function with respect to its input, 
    given its output and the derivative on the output.
    """
    nonlinear_.dreclin_(output,doutput,dinput)

def softplus(input,output):
    """
    Computes the softplus function softplus(input) = log(1+exp(input))
    """
    nonlinear_.softplus_(input,output)

def softmax(input,output):
    """
    Computes the softmax function softmax(input) = exp(input)/sum(exp(input)) = output.
    """
    nonlinear_.softmax_vec_(input,output)

