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

import numpy as np
import scipy.weave
import scipy.linalg
import linalg_

def product_matrix_vector(A,b,x):
    """
    Computes the matrix/vector product A*b=x
    """
    #linalg_.product_matrix_vector_(A,b,x)
    x[:] = np.dot(A,b)

def product_matrix_matrix(A,B,X):
    """
    Computes the matrix/matrix product A*B = X
    """
    #linalg_.product_matrix_matrix_(A,B,X)
    X[:] = np.dot(A,B)

def outer(a,b,X):
    """
    Computes outer product a*b^T=X
    """
    #linalg_.product_matrix_matrix_(np.reshape(a,(-1,1)),np.reshape(b,(1,-1)),X)
    X[:] = np.dot(np.reshape(a,(-1,1)),np.reshape(b,(1,-1)))

def sum_rows(A,x):
    """
    Sums out the rows of A, and puts the result in x
    """
    A.sum(1,out=x)

def sum_columns(A,x):
    """
    Sums out the columns of A, and puts the result in x
    """
    A.sum(0,out=x)

def getdiag(A,x):
    """
    Copies the diagonal of A in x
    """
    #linalg_.getdiag_(A,x)
    x[:] = np.diag(A)

def setdiag(A,x):
    """
    Sets the diagonal of A to x
    """
    #linalg_.setdiag_(A,x)
    for i in range(min(A.shape)):
        A[i,i] = x[i]

def solve(A,B,X,Af=None,Bf=None,pivots=None):
    """
    Solves the linear system A*X = B. If provided,
    will use temporary variables Af, Bf (Fortran ordered double matrix arrays) 
    and pivots (Fortran ordered integer vector array) and avoid memory allocations.
    """
    #if len(A.shape) != 2 or len(B.shape) != 2 or len(X.shape) != 2:
    #    raise ValueError, 'In solve: A, B and X should be matrices'
    #if A.shape[0] != B.shape[0]:
    #    raise ValueError, 'In solve: inputs have incompatible sizes'
    #if A.shape[1] != X.shape[0] or B.shape[1] != X.shape[1]:
    #    raise ValueError, 'In solve: target has incompatible size'
    #
    #if Af is None:
    #   Af = np.array(A,dtype='double',order='fortran')
    #else: 
    #   Af[:] = A
    #
    #if Bf is None:
    #   Bf = np.array(B,dtype='double',order='fortran')
    #else:
    #   Bf[:] = B
    #    
    #if pivots is None:
    #   pivots = np.zeros((A.shape[0]),dtype='i',order='fortran')
    #if len(pivots.shape)!= 1 or pivots.shape[0] != A.shape[0]:
    #   raise ValueError, 'In solve: pivots is not of the right shape'
    #
    #linalg_.solve_(Af,Bf,pivots)
    #X[:] = Bf
    X[:] = np.linalg.solve(A,B)

def lu(A,p,L,U,Af=None,pivots=None):
    """
    Compute the LU decomposition of A[p,:] = L*U, where p is a vector of integers
    and permutes the rows of A.
    If provided, will use temporary variables Af (Fortran ordered double matrix arrays) 
    and pivots (Fortran ordered integer vector array) and avoid memory allocations.
    """
    #if len(A.shape) != 2 or len(L.shape) != 2 or len(U.shape) != 2:
    #    raise ValueError, 'In lu: A, L and U should be matrices'
    #if len(p.shape) != 1:
    #    raise ValueError, 'In lu: p should be a vector'
    #if A.shape[0] != p.shape[0] or \
    #   A.shape[0] != L.shape[0] or A.shape[1] != U.shape[1] or \
    #   L.shape[1] != U.shape[0]:
    #    raise ValueError, 'In lu: A, p, L and U have incompatible sizes'
    #
    #if Af is None:    
    #    Af = np.array(A,dtype='double',order='fortran')
    #else: 
    #    Af[:] = A
    #    
    #if pivots is None:
    #    pivots = np.zeros((min(A.shape)),dtype='i',order='fortran')
    #
    #if len(pivots.shape)!= 1 or pivots.shape[0] != min(A.shape):
    #   raise ValueError, 'In lu: pivots is not of the right shape'
    #    
    #linalg_.lu_(Af,pivots, p, L, U)

    P,Ltemp,Utemp= scipy.linalg.lu(A)
    L[:] = Ltemp
    U[:] = Utemp
    p[:] = np.dot(P.T,np.array(range(P.shape[0])))
