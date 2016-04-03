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
The package ``learners.third_party.libsvm`` contains modules for learning
algorithms using the LIBSVM library. These modules all require that the
LIBSVM library be installed.

To use LIBSVM through mlpython, do the following:

1. download LIBSVM from here: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
2. install LIBSVM (see LIBSVM instructions)
3. install the included python interface (see LIBSVM intrusctions)
4. put path to the python interface in PYTHONPATH

That should do it. Try 'python test.py' to see if your installation is working.

Here is an example of what steps 1 to 3 can look like, where LIBSVMDIR
is the path where you wish to install LIBSVM and
PYTHON_INCLUDEDIR is the path of your python include directory
(use at your own risk!): ::

   tcsh
   set LIBSVMDIR=~/python
   set PYTHON_INCLUDEDIR=/usr/include/python2.6
   cd $LIBSVMDIR/
   wget "http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz"
   tar -zxf libsvm-3.1.tar.gz
   cd libsvm-3.1
   make
   cd python
   make
   exit

Finally, you'll need to add $MLIBSVMDIR/python to your PYTHONPATH.

Currently, ``learner.third_party.libsvm`` contains the following modules:

* ``learning.third_party.libsvm.classification``:    SVM classifier based on the LIBSVM library.

"""
