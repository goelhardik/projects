# Copyright 2014 Hugo Larochelle. All rights reserved.
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

import numpy as np
import mlpython.mathutils.nonlinear as nonlinear

def test_sigmoid():
    """
    Testing nonlinear sigmoid.
    """

    input = np.random.randn(30,20)
    output = np.zeros((30,20))
    nonlinear.sigmoid(input,output)
    assert np.sum(np.abs(output-1/(1+np.exp(-input)))) < 1e-12

def test_dsigmoid():
    """
    Testing nonlinear sigmoid deriv.
    """
    
    input = np.random.randn(30,20)
    output = np.zeros((30,20))
    nonlinear.sigmoid(input,output)
    dinput = np.zeros((30,20))
    doutput = np.random.randn(30,20)
    nonlinear.dsigmoid(output,doutput,dinput)
    assert np.sum(np.abs(dinput-doutput*output*(1-output))) < 1e-12
    
def test_softmax():
    """
    Testing nonlinear softmax.
    """

    input = np.random.randn(20)
    output = np.zeros((20))
    nonlinear.softmax(input,output)
    assert np.sum(np.abs(output-np.exp(input)/np.sum(np.exp(input)))) < 1e-12
    
def test_softplus():
    """
    Testing nonlinear softplus.
    """

    input = np.random.randn(20)
    output = np.zeros((20))
    nonlinear.softplus(input,output)
    print 'NumPy vs mathutils.nonlinear diff. output:',np.sum(np.abs(output-np.log(1+np.exp(input))))
    
    print 'Testing nonlinear reclin'
    input = np.random.randn(30,20)
    output = np.zeros((30,20))
    nonlinear.reclin(input,output)
    print 'NumPy vs mathutils.nonlinear diff. output:',np.sum(np.abs(output-(input>0)*input))
    
    print 'Testing nonlinear reclin deriv.'
    dinput = np.zeros((30,20))
    doutput = np.random.randn(30,20)
    nonlinear.dreclin(output,doutput,dinput)
    print 'NumPy vs mathutils.nonlinear diff. output:',np.sum(np.abs(dinput-(input>0)*doutput))
