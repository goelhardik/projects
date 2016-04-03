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
import scipy.linalg
import mlpython.mathutils.linalg as linalg

def test_product_matrix_vector():
    """
    Test product_matrix_vector agains Numpy.
    """

    A = np.random.rand(30,20)
    b = np.random.rand(10)
    c = np.zeros((40))
    linalg.product_matrix_vector(A[:10,:].T,b,c[10:30])
    assert sum(np.abs(c[10:30] - np.dot(A[:10,:].T,b))) < 1e-10
    A = np.random.rand(20,30)
    b = np.random.rand(30)
    c = np.zeros((20))
    linalg.product_matrix_vector(A,b,c)
    assert sum(np.abs(c - np.dot(A,b))) < 1e-10
    A = np.random.rand(30,20)
    b = np.random.rand(20)
    c = np.zeros((30))
    linalg.product_matrix_vector(A,b,c)
    assert sum(np.abs(c - np.dot(A,b))) < 1e-10

def test_product_matrix_matrix():
    """
    Testing product_matrix_matrix
    """

    A = np.random.rand(20,30)
    B = np.random.rand(30,40)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A,B,C)
    assert np.sum(np.abs(C - np.dot(A,B))) < 1e-10
    A = np.random.rand(30,20)
    B = np.random.rand(30,40)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A.T,B,C)
    np.dot(A.T,B)
    assert np.sum(np.abs(C - np.dot(A.T,B))) < 1e-10
    A = np.random.rand(20,30)
    B = np.random.rand(40,30)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A,B.T,C)
    assert np.sum(np.abs(C - np.dot(A,B.T))) < 1e-10
    A = np.random.rand(30,20)
    B = np.random.rand(40,30)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A.T,B.T,C)
    assert np.sum(np.abs(C - np.dot(A.T,B.T))) < 1e-10
    A = np.random.rand(20,30)
    B = np.random.rand(30,40)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A,B,C.T)
    assert np.sum(np.abs(C.T - np.dot(A,B))) < 1e-10
    A = np.random.rand(30,20)
    B = np.random.rand(30,40)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A.T,B,C.T)
    assert np.sum(np.abs(C.T - np.dot(A.T,B))) < 1e-10
    A = np.random.rand(20,30)
    B = np.random.rand(40,30)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A,B.T,C.T)
    assert np.sum(np.abs(C.T - np.dot(A,B.T))) < 1e-10
    A = np.random.rand(30,20)
    B = np.random.rand(40,30)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A.T,B.T,C.T)
    assert np.sum(np.abs(C.T - np.dot(A.T,B.T))) < 1e-10

    # Testing with fortran order
    A = np.zeros((20,30),order='fortran')
    A[:] = np.random.rand(20,30)
    B = np.random.rand(30,40)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A,B,C)
    assert np.sum(np.abs(C - np.dot(A,B))) < 1e-10
    A = np.zeros((30,20),order='fortran')
    A[:] = np.random.rand(30,20)
    B = np.random.rand(30,40)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A.T,B,C)
    np.dot(A.T,B)
    assert np.sum(np.abs(C - np.dot(A.T,B))) < 1e-10
    A = np.zeros((20,30),order='fortran')
    A[:] = np.random.rand(20,30)
    B = np.random.rand(40,30)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A,B.T,C)
    assert np.sum(np.abs(C - np.dot(A,B.T))) < 1e-10
    A = np.zeros((30,20),order='fortran')
    A[:] = np.random.rand(30,20)
    B = np.random.rand(40,30)
    C = np.zeros((20,40))
    linalg.product_matrix_matrix(A.T,B.T,C)
    assert np.sum(np.abs(C - np.dot(A.T,B.T))) < 1e-10
    A = np.zeros((20,30),order='fortran')
    A[:] = np.random.rand(20,30)
    B = np.random.rand(30,40)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A,B,C.T)
    assert np.sum(np.abs(C.T - np.dot(A,B))) < 1e-10
    A = np.zeros((30,20),order='fortran')
    A[:] = np.random.rand(30,20)
    B = np.random.rand(30,40)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A.T,B,C.T)
    assert np.sum(np.abs(C.T - np.dot(A.T,B))) < 1e-10
    A = np.zeros((20,30),order='fortran')
    A[:] = np.random.rand(20,30)
    B = np.random.rand(40,30)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A,B.T,C.T)
    assert np.sum(np.abs(C.T - np.dot(A,B.T))) < 1e-10
    A = np.zeros((30,20),order='fortran')
    A[:] = np.random.rand(30,20)
    B = np.random.rand(40,30)
    C = np.zeros((40,20))
    linalg.product_matrix_matrix(A.T,B.T,C.T)
    assert np.sum(np.abs(C.T - np.dot(A.T,B.T))) < 1e-10

    A = np.random.rand(1,30)
    B = np.random.rand(30,1)
    C = np.zeros((1,1))
    linalg.product_matrix_matrix(A,B,C)
    assert np.sum(np.abs(C - np.dot(A,B))) < 1e-10

def test_getdiag():
    """
    Testing getdiag.
    """

    A = np.random.rand(30,20)
    x = np.zeros((20))
    linalg.getdiag(A,x)
    assert np.sum(np.abs(x-np.diag(A))) < 1e-10
    A = np.random.rand(30,20).T
    x = np.zeros((20))
    linalg.getdiag(A,x)
    assert np.sum(np.abs(x-np.diag(A))) < 1e-10

def test_setdiag():
    """
    Testing setdiag.
    """

    A = np.random.rand(30,20)
    x = np.random.rand(20)
    linalg.setdiag(A,x)
    assert np.sum(np.abs(x-np.diag(A))) < 1e-10
    A = np.random.rand(30,20).T
    x = np.random.rand(20)
    linalg.setdiag(A,x)
    assert np.sum(np.abs(x-np.diag(A))) < 1e-10

def test_solve():
    """
    Testing solve.
    """

    A = np.random.rand(30,30)
    A = np.dot(A,A.T)
    B = np.random.rand(30,20)
    X = np.zeros((30,20))
    linalg.solve(A,B,X)
    assert np.mean(np.abs(X-np.linalg.solve(A,B))) < 1e-6
    A = np.random.rand(30,30)
    A = np.dot(A,A.T)
    B = np.random.rand(30,30)
    X = np.zeros((30,30))
    linalg.solve(A,B,X)
    assert np.mean(np.abs(X-np.linalg.solve(A,B))) < 1e-6

def test_lu():
    """
    Testing lu decomposition.
    """

    A = np.random.rand(4,6)
    p = np.zeros((4),dtype='i')
    L = np.zeros((4,4))
    U = np.zeros((4,6))
    linalg.lu(A,p,L,U)

    # Writing permutation vector p in matrix form
    P = np.zeros((4,4))
    for P_row,p_el in zip(P.T,p):
        P_row[p_el] = 1
    P2,L2,U2 = scipy.linalg.lu(A)
    assert np.sum(np.abs(P-P2)) < 1e-10
    assert np.sum(np.abs(L-L2)) < 1e-10
    assert np.sum(np.abs(U-U2)) < 1e-10
    
    A = np.random.rand(20,30)
    p = np.zeros((20),dtype='i')
    L = np.zeros((20,20))
    U = np.zeros((20,30))
    linalg.lu(A,p,L,U)

    # Writing permutation vector p in matrix form
    P = np.zeros((20,20))
    for P_row,p_el in zip(P.T,p):
        P_row[p_el] = 1
    P2,L2,U2 = scipy.linalg.lu(A)
    assert np.sum(np.abs(P-P2)) < 1e-10
    assert np.sum(np.abs(L-L2)) < 1e-10
    assert np.sum(np.abs(U-U2)) < 1e-10
    
    A = np.random.rand(30,20)
    p = np.zeros((30),dtype='i')
    L = np.zeros((30,20))
    U = np.zeros((20,20))
    linalg.lu(A,p,L,U)

    # Writing permutation vector p in matrix form
    P = np.zeros((30,30))
    for P_row,p_el in zip(P.T,p):
        P_row[p_el] = 1
    P2,L2,U2 = scipy.linalg.lu(A)
    assert np.sum(np.abs(P-P2)) < 1e-10
    assert np.sum(np.abs(L-L2)) < 1e-10
    assert np.sum(np.abs(U-U2)) < 1e-10

