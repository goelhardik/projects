// Copyright 2011 Hugo Larochelle. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 
//    1. Redistributions of source code must retain the above copyright notice, this list of
//       conditions and the following disclaimer.
// 
//    2. Redistributions in binary form must reproduce the above copyright notice, this list
//       of conditions and the following disclaimer in the documentation and/or other materials
//       provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// The views and conclusions contained in the software and documentation are those of the
// authors and should not be interpreted as representing official policies, either expressed
// or implied, of Hugo Larochelle.

#include "Python.h"
#include "numpy/arrayobject.h"
#include "linalg_.h"

/* This file interfaces with useful non-Python routines. 
   To use them however, see linalg.py. */

static PyMethodDef linalg_[] = {
  {"product_matrix_vector_", product_matrix_vector_, METH_VARARGS, "Interface to dgemv BLAS function. All arguments should be contiguous arrays of doubles."},
  {"product_matrix_matrix_", product_matrix_matrix_, METH_VARARGS, "Interface to dgemm BLAS function. All arguments should be contiguous arrays of doubles."},
  {"getdiag_", getdiag_, METH_VARARGS, "Copies the diagonal of an input matrix in a target vector."},
  {"setdiag_", setdiag_, METH_VARARGS, "Set the diagonal of an input matrix to the value of an input vector."},
  {"solve_", solve_, METH_VARARGS, "Interface to dgesv LAPACK function. Requires Fortran-contiguous Numpy arrays as input."},
  {"lu_", lu_, METH_VARARGS, "Interface to dgetrf LAPACK function. Requires Fortran-contiguous Numpy arrays as input (the A and pivot arrays). The p, L and U targets should be contiguous, but not necessarly Fortran-ordered."},
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

PyMODINIT_FUNC
initlinalg_()  {
  (void) Py_InitModule("linalg_", linalg_);
  import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *product_matrix_vector_(PyObject *self, PyObject *args)
{
  PyArrayObject *matrix, *vector, *result;
  double alpha = 1.;
  double beta = 0.;
  int int_one = 1;
  int m, n;
  char trans[2] = "T";

  extern void dgemv_(char *trans, int *m, int *n,
                     double *alpha, double *a, int *lda,
                     double *x, int *incx,
                     double *beta, double *Y, int *incy);
 
  if (!PyArg_ParseTuple(args, "O!O!O!", 
                        &PyArray_Type, &matrix, 
                        &PyArray_Type, &vector, 
                        &PyArray_Type, &result)) return NULL;

  if ( (NULL == matrix) || (NULL == vector) || (NULL == result) ) return NULL;

  if ( (matrix->descr->type_num != NPY_DOUBLE) || 
       (vector->descr->type_num != NPY_DOUBLE) ||
       (result->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(matrix,NPY_ALIGNED) ||
       !(PyArray_CHKFLAGS(matrix,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(matrix,NPY_F_CONTIGUOUS)) ||
       !PyArray_CHKFLAGS(vector,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(result,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_vector: all arguments must be of type double, contiguous and aligned, and targets should be writeable");
    return NULL;
  }

  if ( (matrix->nd != 2) || (vector->nd != 1) || (result->nd != 1) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_vector: not all arguments have the right dimensionality");
    return NULL;
  }

  /* Figure out correct values for args to blas*/
  if (PyArray_ISFORTRAN(matrix)) {
    m = matrix->dimensions[0];
    n = matrix->dimensions[1];
    trans[0] = 'N';
  }
  else /*if (PyArray_ISCONTIGUOUS(matrix))*/ {
    /* Equivalent to matrix being the transposed of a Fortran matrix*/
    m = matrix->dimensions[1];
    n = matrix->dimensions[0];
  }

  /* Check if dimensions are compatible */
  if (matrix->dimensions[1] != vector->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_vector: input dimensions are not compatible");
    return NULL;
  }

  if (result->dimensions[0] != matrix->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_vector: target dimension is not compatible");
    return NULL;
  }
 
  dgemv_(trans, &m, &n, &alpha, (double *)matrix->data, &m,
         (double *)vector->data, &int_one,
         &beta, (double *)result->data, &int_one);
  Py_RETURN_NONE;
}

static PyObject *product_matrix_matrix_(PyObject *self, PyObject *args)
{
  PyArrayObject *matrix1, *matrix2, *result, *A, *B;
  double alpha = 1.;
  double beta = 0.;
  int m, n, k; /* Watch out: m and n have a different meaning from that for dgemv, where
                             it is the dimensionality BEFORE the transpose, whereas
                             dgemm requires the dimensionality AFTER. */
  int lda, ldb, ldc;
  char transa[2] = "T";
  char transb[2] = "T";
 
  extern void dgemm_(char* transa, char* transb, int* m, 
                     int* n, int* k, double *alpha, 
                     double *a, int* lda, double *b, 
                     int *ldb, double *beta, double *c, int* ldc);

  if (!PyArg_ParseTuple(args, "O!O!O!", 
                        &PyArray_Type, &matrix1,
                        &PyArray_Type, &matrix2, 
                        &PyArray_Type, &result)) return NULL;

  if ( (NULL == matrix1) || (NULL == matrix2) || (NULL == result) ) return NULL;

  if ( (matrix1->descr->type_num != NPY_DOUBLE) || 
       (matrix2->descr->type_num != NPY_DOUBLE) ||
       (result->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(matrix1,NPY_ALIGNED) ||
       !(PyArray_CHKFLAGS(matrix1,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(matrix1,NPY_F_CONTIGUOUS)) ||
       !PyArray_CHKFLAGS(matrix2,NPY_ALIGNED) ||
       !(PyArray_CHKFLAGS(matrix2,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(matrix2,NPY_F_CONTIGUOUS)) ||
       !PyArray_CHKFLAGS(result,NPY_ALIGNED|NPY_WRITEABLE) ||
       !(PyArray_CHKFLAGS(result,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(result,NPY_F_CONTIGUOUS)) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_matrix: all arguments must be of type double, contiguous and aligned, and targets should be writeable");
    return NULL;
  }

  if ( (matrix1->nd != 2) || (matrix2->nd != 2) || (result->nd != 2) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_matrix: not all arguments have the right dimensionality");
    return NULL;
  }
  
  /* Figure out correct values for args to blas*/
  if (PyArray_ISFORTRAN(result)) {
    /* Matrix result has Fortran order */
    A = matrix1;
    B = matrix2;

    m = matrix1->dimensions[0];
    k = matrix1->dimensions[1];
    if (PyArray_ISFORTRAN(matrix1)) {
      transa[0] = 'N';
      lda = m;
    }
    else /*if (PyArray_ISCONTIGUOUS(matrix1))*/ {
      /* Equivalent to matrix being the transposed of a Fortran matrix*/
      lda = k;
    }
    
    n = matrix2->dimensions[1];
    k = matrix2->dimensions[0];
    if (PyArray_ISFORTRAN(matrix2)) {
      transb[0] = 'N';
      ldb = k;
    }
    else /*if (PyArray_ISCONTIGUOUS(matrix2))*/ {
      /* Equivalent to matrix being the transposed of a Fortran matrix*/
      ldb = n;
    }
  }
  else /*if (PyArray_ISCONTIGUOUS(result))*/ {
    /* Matrix result has C order! Must be careful how blas is called!
       Must interchange matrix1 with matrix2, and interchange 
       how the C and Fortran matrices are processed. */
    A = matrix2;
    B = matrix1;

    m = matrix2->dimensions[1];
    k = matrix2->dimensions[0];
    if (PyArray_ISFORTRAN(matrix2)) {
      lda = k;
    }
    else /*if (PyArray_ISCONTIGUOUS(matrix2))*/ {
      /* Equivalent to matrix being the transposed of a Fortran matrix*/
      transa[0] = 'N';
      lda = m;
    }

    n = matrix1->dimensions[0];
    k = matrix1->dimensions[1];
    if (PyArray_ISFORTRAN(matrix1)) {
      ldb = n;
    }
    else /*if (PyArray_ISCONTIGUOUS(matrix1))*/ {
      /* Equivalent to matrix being the transposed of a Fortran matrix*/
      transb[0] = 'N';
      ldb = k;
    }
  }
  ldc = m;

  if ( matrix1->dimensions[1] != matrix2->dimensions[0] ) {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_matrix: input matrices dimensions are not compatible");
    return NULL;
  }

  if ( (matrix1->dimensions[0] != result->dimensions[0]) || 
       (matrix2->dimensions[1] != result->dimensions[1]) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In product_matrix_matrix: target dimensions are not compatible");
    return NULL;
  }

  dgemm_(transa, transb, &m, &n, &k, &alpha, (double *)A->data, &lda,
         (double *)B->data, &ldb, &beta,
         (double *)result->data, &ldc );
  
  Py_RETURN_NONE;
}

static PyObject *getdiag_(PyObject *self, PyObject *args)
{
  PyArrayObject *matrix, *result;
  int m, n;

  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyArray_Type, &matrix,
                        &PyArray_Type, &result)) return NULL;

  if ( (NULL == matrix) || (NULL == result) ) return NULL;

  if ( (matrix->descr->type_num != NPY_DOUBLE) || 
       (result->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(matrix,NPY_ALIGNED) ||
       !(PyArray_CHKFLAGS(matrix,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(matrix,NPY_F_CONTIGUOUS)) ||
       !PyArray_CHKFLAGS(result,NPY_C_CONTIGUOUS||NPY_ALIGNED|NPY_WRITEABLE)) {
    PyErr_SetString(PyExc_ValueError,
                    "In getdiag: some arguments are of invalid type");
    return NULL;
  }
  
  if ( (matrix->nd != 2) || (result->nd != 1) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In getdiag: not all arguments have the right dimensionality");
    return NULL;
  }
  
  m = matrix->dimensions[0];
  n = matrix->dimensions[1];
  int diagsize = (m<n?m:n);
  if ( diagsize != result->dimensions[0] ) {
    PyErr_SetString(PyExc_ValueError,
                    "In getdiag: input matrix and target vector dimensions are not compatible");
    return NULL;
  }

  char * matrix_data_iter = matrix->data;
  int s1 = matrix->strides[0];
  int s2 = matrix->strides[1];
  double * result_data_iter = (double *) result->data;
  int i;
  for (i=0; i<diagsize; i++)
  {
    *result_data_iter++ = *((double *)matrix_data_iter);
    matrix_data_iter += s1 + s2;
  }
  Py_RETURN_NONE;
}

static PyObject *setdiag_(PyObject *self, PyObject *args)
{
  PyArrayObject *matrix, *diag;
  int m, n;

  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyArray_Type, &matrix,
                        &PyArray_Type, &diag)) return NULL;

  if ( (NULL == matrix) || (NULL == diag) ) return NULL;

  if ( (matrix->descr->type_num != NPY_DOUBLE) || 
       (diag->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(matrix,NPY_ALIGNED|NPY_WRITEABLE) ||
       !(PyArray_CHKFLAGS(matrix,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(matrix,NPY_F_CONTIGUOUS)) ||
       !PyArray_CHKFLAGS(diag,NPY_C_CONTIGUOUS||NPY_ALIGNED)) {
    PyErr_SetString(PyExc_ValueError,
                    "In setdiag: some arguments are of invalid type");
    return NULL;
  }
  
  if ( (matrix->nd != 2) || (diag->nd != 1) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In setdiag: not all arguments have the right dimensionality");
    return NULL;
  }
  
  m = matrix->dimensions[0];
  n = matrix->dimensions[1];
  int diagsize = (m<n?m:n);
  if ( diagsize != diag->dimensions[0] ) {
    PyErr_SetString(PyExc_ValueError,
                    "In setdiag: input matrix and target vector dimensions are not compatible");
    return NULL;
  }

  char * matrix_data_iter = matrix->data;
  int s1 = matrix->strides[0];
  int s2 = matrix->strides[1];
  double * diag_data_iter = (double *) diag->data;
  int i;
  for (i=0; i<diagsize; i++)
  {
    *((double *)matrix_data_iter) = *diag_data_iter++;
    matrix_data_iter += s1 + s2;
  }
  Py_RETURN_NONE;
}

static PyObject *solve_(PyObject *self, PyObject *args)
{
  PyArrayObject *A, *B, *pivots;
  int n, nrhs, lda, ldb, info = 0;

  extern void dgesv_(int* n, int* nrhs, double * A,
                     int* lda, int* ipiv, double *B, 
                     int *ldb, int* info);

  if (!PyArg_ParseTuple(args, "O!O!O!", 
                        &PyArray_Type, &A,
                        &PyArray_Type, &B, 
                        &PyArray_Type, &pivots)) return NULL;

  if ( (NULL == A) || (NULL == B) || (NULL == pivots) ) return NULL;
  if ( (A->descr->type_num != NPY_DOUBLE) || 
       (B->descr->type_num != NPY_DOUBLE) ||
       (pivots->descr->type_num != NPY_INT) ||
       !PyArray_CHKFLAGS(A,NPY_F_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ||
       !PyArray_CHKFLAGS(B,NPY_F_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ||
       !PyArray_CHKFLAGS(pivots,NPY_F_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In solve: some arguments are of invalid type");
    return NULL;
  }
  
  n = A->dimensions[0];
  nrhs = B->dimensions[1];
  lda = n;
  ldb = B->dimensions[0];

  dgesv_(&n, &nrhs, (double *) A->data, &lda, (int *) pivots->data, 
	 (double *) B->data, &ldb, &info);
  
  if ( info < 0 )
  {
    /* Argument "-info" has illegal value */
    PyErr_SetString(PyExc_ValueError,
                    "In solve: dgesv error, one of the arguments has illegal value");
    return NULL;      
  }
  else if (info > 0 )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In solve: dgesv error, solution could not be computed for value of A");
    return NULL;            
  }
  
  Py_RETURN_NONE;
}

static PyObject *lu_(PyObject *self, PyObject *args) 
{
  PyArrayObject *A, *pivots, *p, *L, *U;
  int m, n, lda, info = 0;

  extern void dgetrf_(int* m, int* n, double * A,
                     int* lda, int* ipiv, int* info);

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!", 
                        &PyArray_Type, &A,
                        &PyArray_Type, &pivots,
                        &PyArray_Type, &p,
                        &PyArray_Type, &L,
                        &PyArray_Type, &U)) return NULL;

  if ( (NULL == A) || (NULL == pivots) ) return NULL;
  if ( (A->descr->type_num != NPY_DOUBLE) || 
       (pivots->descr->type_num != NPY_INT) ||
       (p->descr->type_num != NPY_INT) || 
       (U->descr->type_num != NPY_DOUBLE) || 
       (L->descr->type_num != NPY_DOUBLE) || 
       !PyArray_CHKFLAGS(A,NPY_F_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ||
       !PyArray_CHKFLAGS(pivots,NPY_F_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ||
       !PyArray_CHKFLAGS(p,NPY_C_CONTIGUOUS) || /* Contiguous vectors are for both C and fortran */
       !(PyArray_CHKFLAGS(L,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(L,NPY_F_CONTIGUOUS)) ||
       !(PyArray_CHKFLAGS(U,NPY_C_CONTIGUOUS) || PyArray_CHKFLAGS(U,NPY_F_CONTIGUOUS)) ||
       !PyArray_CHKFLAGS(p,NPY_ALIGNED|NPY_WRITEABLE) ||
       !PyArray_CHKFLAGS(L,NPY_ALIGNED|NPY_WRITEABLE) ||
       !PyArray_CHKFLAGS(U,NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In lu: some arguments are of invalid type");
    return NULL;
  }
  
  m = A->dimensions[0];
  n = A->dimensions[1];
  lda = m;

  dgetrf_(&m, &n, (double *) A->data, &lda, (int *) pivots->data, &info);
       
  if ( info < 0 )
  {
    /* Argument "-info" has illegal value */
    PyErr_SetString(PyExc_ValueError,
                    "In lu: dgetrf error, one of the arguments has illegal value");
    return NULL;      
  }
  else if (info > 0 )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In lu: dgetrf error, solution could not be computed for value of A");
    return NULL;            
  }
       
  /* Rearrange the pivots in the right form*/
  int * pivots_ptr, *p_ptr;
  int npivots = (m<n?m:n);
  int temp;
  int i, j;
  pivots_ptr = (int *)pivots->data;
  p_ptr = (int *)p->data;
  /* Init p */
  for ( i=0; i<m; i++ )
    *p_ptr++ = i;

  p_ptr = (int *)p->data;
  for ( i=0; i<npivots; i++ )
  {
    temp = p_ptr[*pivots_ptr-1];
    p_ptr[*pivots_ptr-1] = p_ptr[i];
    p_ptr[i] = temp;
    pivots_ptr++;
  }
  /* Compute L */
  double *A_ptr, *L_ptr;
  int Asi = A->strides[0];
  int Asj = A->strides[1];
  int Lsi = L->strides[0];
  int Lsj = L->strides[1];
  for ( i=0; i<m; i++ )
    for ( j=0; j<npivots; j++ )
    {
      A_ptr = (double *)(A->data + i*Asi + j*Asj);
      L_ptr = (double *)(L->data + i*Lsi + j*Lsj);
      if ( j < i ) {
        *L_ptr = *A_ptr; 
      }
      else if ( j > i )
      {
        *L_ptr = 0.;
      }
      else /* if ( j == i ) */
      {
        *L_ptr = 1.;
      }
    }

  double *U_ptr;
  int Usi = U->strides[0];
  int Usj = U->strides[1];
  for ( i=0; i<npivots; i++ )
    for ( j=0; j<n; j++ )
    {
      A_ptr = (double *)(A->data + i*Asi + j*Asj);
      U_ptr = (double *)(U->data + i*Usi + j*Usj);
      if ( j < i ) {
        *U_ptr = 0.; 
      }
      else 
      {
        *U_ptr = *A_ptr; 
      }
    }

  Py_RETURN_NONE;
}
