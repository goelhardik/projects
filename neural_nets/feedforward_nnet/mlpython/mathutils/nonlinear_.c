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
#include "math.h"
#include "numpy/arrayobject.h"
#include "nonlinear_.h"

/* This file interfaces with useful non-Python routines. 
   To use them however, see nonlinear.py. */

static PyMethodDef nonlinear_[] = {
  {"sigmoid_", sigmoid_, METH_VARARGS, "Computes the sigmoid function."},
  {"dsigmoid_", dsigmoid_, METH_VARARGS, "Computes the derivative of the sigmoid function."},
  {"softmax_vec_", softmax_vec_, METH_VARARGS, "Computes the softmax function for an input vector."},
  {"reclin_", reclin_, METH_VARARGS, "Computes the rectified linear function."},
  {"dreclin_", dreclin_, METH_VARARGS, "Computes the derivative of the rectified linear function."},
  {"softplus_", softplus_, METH_VARARGS, "Computes the softplus function."},
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

PyMODINIT_FUNC
initnonlinear_()  {
  (void) Py_InitModule("nonlinear_", nonlinear_);
  import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *sigmoid_(PyObject *self, PyObject *args)
{
  PyArrayObject *input, *output;
  int i;

  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyArray_Type, &input, 
                        &PyArray_Type, &output)) return NULL;

  if ( (NULL == input) || (NULL == output) ) return NULL;
  
  if ( (input->descr->type_num != NPY_DOUBLE) || 
       (output->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(input,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(output,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In sigmoid_: all arguments must be of type double, C contiguous and aligned, and output should be writeable");
    return NULL;
  }
      
  if ( (input->nd != output->nd) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In sigmoid_: both arguments should have the same dimensionality");
    return NULL;
  }

  int tot_dim = 1;
  for (i=0; i<input->nd; i++)
  {
    if ( (input->dimensions[i] != output->dimensions[i]) )
    {
      PyErr_SetString(PyExc_ValueError,
		      "In sigmoid_: all dimensions of both arguments should be equal");
      return NULL;
    }
    tot_dim *= input->dimensions[i];
  }
  
  double * input_data_iter = (double *) input->data;
  double * output_data_iter = (double *) output->data;
  for (i=0; i<tot_dim; i++)
  {
    *output_data_iter++ = 1./(1.+exp(-*input_data_iter++));
  }
  Py_RETURN_NONE;
}

static PyObject *dsigmoid_(PyObject *self, PyObject *args)
{
  PyArrayObject *output,*doutput,*dinput;
  int i;

  if (!PyArg_ParseTuple(args, "O!O!O!", 
                        &PyArray_Type, &output, 
                        &PyArray_Type, &doutput,
                        &PyArray_Type, &dinput)) return NULL;

  if ( (NULL == output) || (NULL == doutput) || (NULL == dinput) ) return NULL;
  
  if ( (output->descr->type_num != NPY_DOUBLE) || 
       (doutput->descr->type_num != NPY_DOUBLE) ||
       (dinput->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(output,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(doutput,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(dinput,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In dsigmoid_: all arguments must be of type double, C contiguous and aligned, and output should be writeable");
    return NULL;
  }
  
  if ( (dinput->nd != output->nd) || (doutput->nd != output->nd))
  {
    PyErr_SetString(PyExc_ValueError,
                    "In dsigmoid_: all arguments should have the same dimensionality");
    return NULL;
  }

  int tot_dim = 1;
  for (i=0; i<output->nd; i++)
  {
    if ( (output->dimensions[i] != doutput->dimensions[i]) || (output->dimensions[i] != dinput->dimensions[i]) )
    {
      PyErr_SetString(PyExc_ValueError,
		      "In dsigmoid_: all dimensions of al arguments should be equal");
      return NULL;
    }
    tot_dim *= output->dimensions[i];
  }
  
  double * output_data_iter = (double *) output->data;
  double * doutput_data_iter = (double *) doutput->data;
  double * dinput_data_iter = (double *) dinput->data;
  for (i=0; i<tot_dim; i++)
  {
    *dinput_data_iter++ = *output_data_iter * (1.-*output_data_iter) * *doutput_data_iter++;
    output_data_iter++;
  }
  Py_RETURN_NONE;
}

static PyObject *softmax_vec_(PyObject *self, PyObject *args)
{
  PyArrayObject *input, *output;
  int i;

  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyArray_Type, &input, 
                        &PyArray_Type, &output)) return NULL;

  if ( (NULL == input) || (NULL == output) ) return NULL;
  
  if ( (input->descr->type_num != NPY_DOUBLE) || 
       (output->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(input,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(output,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In softmax_vec_: all arguments must be of type double, C contiguous and aligned, and output should be writeable");
    return NULL;
  }
      
  if ( (input->nd != output->nd) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In softmax_vec_: both arguments should have the same dimensionality");
    return NULL;
  }

  int tot_dim = 1;
  for (i=0; i<input->nd; i++)
  {
    if ( (input->dimensions[i] != output->dimensions[i]) )
    {
      PyErr_SetString(PyExc_ValueError,
		      "In softmax_vec_: all dimensions of both arguments should be equal");
      return NULL;
    }
    tot_dim *= input->dimensions[i];
  }
  
  double * input_data_iter = (double *) input->data;
  double * output_data_iter = (double *) output->data;
  double max = *input_data_iter;
  for (i=0; i<tot_dim; i++)
  {
    if (max < *input_data_iter){ max = *input_data_iter; }
    input_data_iter++;
  }

  input_data_iter = (double *) input->data;
  output_data_iter = (double *) output->data;
  double sum = 0;
  for (i=0; i<tot_dim; i++)
  {
    *output_data_iter = exp(*input_data_iter++ - max);
    sum += *output_data_iter++;
  }

  output_data_iter = (double *) output->data;
  for (i=0; i<tot_dim; i++)
  {
    *output_data_iter++ /= sum;
  }


  Py_RETURN_NONE;
}

static PyObject *reclin_(PyObject *self, PyObject *args)
{
  PyArrayObject *input, *output;
  int i;

  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyArray_Type, &input, 
                        &PyArray_Type, &output)) return NULL;

  if ( (NULL == input) || (NULL == output) ) return NULL;
  
  if ( (input->descr->type_num != NPY_DOUBLE) || 
       (output->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(input,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(output,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In reclin_: all arguments must be of type double, C contiguous and aligned, and output should be writeable");
    return NULL;
  }
      
  if ( (input->nd != output->nd) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In reclin_: both arguments should have the same dimensionality");
    return NULL;
  }

  int tot_dim = 1;
  for (i=0; i<input->nd; i++)
  {
    if ( (input->dimensions[i] != output->dimensions[i]) )
    {
      PyErr_SetString(PyExc_ValueError,
		      "In reclin_: all dimensions of both arguments should be equal");
      return NULL;
    }
    tot_dim *= input->dimensions[i];
  }
  
  double * input_data_iter = (double *) input->data;
  double * output_data_iter = (double *) output->data;
  double input_data;
  for (i=0; i<tot_dim; i++)
  {
    input_data = *input_data_iter++;
    if(input_data <= 0)
      *output_data_iter++ = 0;
    else
      *output_data_iter++ = input_data;
  }
  Py_RETURN_NONE;
}

static PyObject *dreclin_(PyObject *self, PyObject *args)
{
  PyArrayObject *output,*doutput,*dinput;
  int i;

  if (!PyArg_ParseTuple(args, "O!O!O!", 
                        &PyArray_Type, &output, 
                        &PyArray_Type, &doutput,
                        &PyArray_Type, &dinput)) return NULL;

  if ( (NULL == output) || (NULL == doutput) || (NULL == dinput) ) return NULL;
  
  if ( (output->descr->type_num != NPY_DOUBLE) || 
       (doutput->descr->type_num != NPY_DOUBLE) ||
       (dinput->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(output,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(doutput,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(dinput,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In dreclin_: all arguments must be of type double, C contiguous and aligned, and output should be writeable");
    return NULL;
  }
  
  if ( (dinput->nd != output->nd) || (doutput->nd != output->nd))
  {
    PyErr_SetString(PyExc_ValueError,
                    "In dreclin_: all arguments should have the same dimensionality");
    return NULL;
  }

  int tot_dim = 1;
  for (i=0; i<output->nd; i++)
  {
    if ( (output->dimensions[i] != doutput->dimensions[i]) || (output->dimensions[i] != dinput->dimensions[i]) )
    {
      PyErr_SetString(PyExc_ValueError,
		      "In dreclin_: all dimensions of al arguments should be equal");
      return NULL;
    }
    tot_dim *= output->dimensions[i];
  }
  
  double * output_data_iter = (double *) output->data;
  double * doutput_data_iter = (double *) doutput->data;
  double * dinput_data_iter = (double *) dinput->data;
  for (i=0; i<tot_dim; i++)
  {
    if (*output_data_iter++<=0)
      *dinput_data_iter++ = 0;
    else
      *dinput_data_iter++ = *doutput_data_iter;
    doutput_data_iter++;
  }
  Py_RETURN_NONE;
}

static PyObject *softplus_(PyObject *self, PyObject *args)
{
  PyArrayObject *input, *output;
  int i;

  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyArray_Type, &input, 
                        &PyArray_Type, &output)) return NULL;

  if ( (NULL == input) || (NULL == output) ) return NULL;
  
  if ( (input->descr->type_num != NPY_DOUBLE) || 
       (output->descr->type_num != NPY_DOUBLE) ||
       !PyArray_CHKFLAGS(input,NPY_C_CONTIGUOUS|NPY_ALIGNED) ||
       !PyArray_CHKFLAGS(output,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE) ) {
    PyErr_SetString(PyExc_ValueError,
                    "In softplus_: all arguments must be of type double, C contiguous and aligned, and output should be writeable");
    return NULL;
  }
      
  if ( (input->nd != output->nd) )
  {
    PyErr_SetString(PyExc_ValueError,
                    "In softplus_: both arguments should have the same dimensionality");
    return NULL;
  }

  int tot_dim = 1;
  for (i=0; i<input->nd; i++)
  {
    if ( (input->dimensions[i] != output->dimensions[i]) )
    {
      PyErr_SetString(PyExc_ValueError,
		      "In softplus_: all dimensions of both arguments should be equal");
      return NULL;
    }
    tot_dim *= input->dimensions[i];
  }
  
  double * input_data_iter = (double *) input->data;
  double * output_data_iter = (double *) output->data;
  double input_data;
  for (i=0; i<tot_dim; i++)
  {
    input_data = *input_data_iter++;
    if(input_data > 0)
      *output_data_iter++ = input_data + log(1+exp(-input_data));
    else
      *output_data_iter++ = log(1+exp(input_data));
//    if(input_data > 35)
//      *output_data_iter++ = input_data;
//    else
//      if(input_data < -40)
//	*output_data_iter++ = 0;
//      else
//	*output_data_iter++ = log(1+exp(input_data));
  }
  Py_RETURN_NONE;
}

