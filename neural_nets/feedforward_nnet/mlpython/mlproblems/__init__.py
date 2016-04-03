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
Before a dataset can be fed to a Learner, it must first be converted
into an MLProblem. 

MLProblem objects are simply iterators with some extra properties.
Hence, from an MLProblem, examples can be obtained by iterating over
the MLProblem. 

MLProblem objects also contain metadata, i.e. "data about the
data". For instance, the metadata could contain information about the
size of the input or the set of all possible values for the
target. The metadata (field ``metadata`` of an MLProblem) is
represented by a dictionary mapping strings to arbitrary objects.  

Finally, an MLProblem has a length, as defined by the output of
``__len__(self)``.

The ``mlproblems`` package is divided into different modules, 
based on the nature of the machine learning problem or task
being implemented. 

The modules are:

* ``mlproblems.generic``:          MLProblems not specific to a particular task.
* ``mlproblems.classification``:   classification MLProblems.
* ``mlproblems.ranking``:          ranking MLProblems.

"""
