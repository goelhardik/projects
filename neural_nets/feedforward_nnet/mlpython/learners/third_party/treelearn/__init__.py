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
The package ``learners.third_party.treelearn`` contains modules for learning
algorithms implemented by in the TreeLearn library. These modules all require that the
TreeLearn and scikits-learn libraries be installed.

To install scikits-learn, one option is to use easy_install: ::

   easy_install -U scikit-learn

For other ways of installing scikits-learn, see http://scikit-learn.sourceforge.net/dev/install.html#installing-an-official-release.

To install TreeLearn:

1. download TreeLearn through this link: https://github.com/capitalk/treelearn/zipball/master
2. unzip the downloaded content and run (possibly with sudo): ::

    python setup.py install

And that should do it!

Currently, ``learner.third_party.treelearn`` contains the following modules:

* ``learning.third_party.treelearn.classification``:    Classifiers from the TreeLearn library.
* ``learning.third_party.treelearn.regression``:        Regression models from the TreeLearn library.

"""
