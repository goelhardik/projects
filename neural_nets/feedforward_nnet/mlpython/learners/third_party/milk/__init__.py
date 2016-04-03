# Copyright 2011 David Brouillard - Guillaume Roy-Fontaine. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY David Brouillard - Guillaume Roy-Fontaine ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL David Brouillard - Guillaume Roy-Fontaine OR
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
The package ``learners.third_party.milk`` contains modules for learning
algorithms using the Milk library. These modules all require that the
Milk library be installed.

To use milk through mlpython, do the following:

1. download milk from here: http://pypi.python.org/pypi/milk/#downloads
2. install milk (instructions from Milk's INSTALL.rst)
   The following should work: ::

      python setup.py install

3. **IMPORTANT** When importing milk for the first time, make sure not to be in its directory.

And that should do it! Try 'import milk' to see if your installation is working.

Currently, ``learner.third_party.milk`` contains the following modules:

* ``learning.third_party.milk.classification``:    Classifiers from the Milk library.

"""
