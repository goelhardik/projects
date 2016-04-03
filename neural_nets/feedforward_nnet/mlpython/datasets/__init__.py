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
The ``datasets`` package provides a common framework for downloading
and loading datasets. It is perfect for someone who wishes to
experiment with a Leaner and wants quick access to many arbitrary datasets.

The package include a module for each currently supported dataset. The
module docstring should give a reference of work that produced the dataset
or that used this particular version of the dataset.

The package also has a ``datasets.store`` module implements simple functions
to obtain MLProblems from those datasets.

The modules currently included are:

* ``datasets.store``:                            provides functions for obtaining MLProblems from the supported datasets.
* ``datasets.abalone``:                          Abalone dataset module.
* ``datasets.adult``:                            Adult dataset module.
* ``datasets.bibtex``:                           Bibtex dataset module.
* ``datasets.binarized_mnist``:                  binarized version of MNIST module.
* ``datasets.cadata``:                           Cadata dataset module.
* ``datasets.cifar10``:                          CIFAR-10 dataset module.
* ``datasets.connect4``:                         Connect-4 dataset module.
* ``datasets.corel5k``:                          Corel5k dataset module.
* ``datasets.corrupted_mnist``:                  Corrupted MNIST dataset module.
* ``datasets.corrupted_ocr_letters``:            Corrupted OCR letters dataset module.
* ``datasets.dna``:                              DNA dataset module.
* ``datasets.face_completion_lfw``:              Labeled Faces in the Wild, face completion dataset module.
* ``datasets.housing``:                          Housing dataset module.
* ``datasets.heart``:                            Heart dataset module.
* ``datasets.letor_mq2007``:                     LETOR 4.0 MQ2007 dataset module.
* ``datasets.letor_mq2008``:                     LETOR 4.0 MQ2008 dataset module.
* ``datasets.majmin``:                           MajMin dataset module.
* ``datasets.mediamill``:                        Mediamill dataset module.
* ``datasets.medical``:                          Medical dataset module.
* ``datasets.mnist``:                            MNIST dataset module.
* ``datasets.mnist_basic``:                      MNIST basic dataset module.
* ``datasets.mnist_background_images``:          MNIST background-images dataset module.
* ``datasets.mnist_background_random``:          MNIST background-random dataset module.
* ``datasets.mnist_rotated``:                    MNIST rotated dataset module.
* ``datasets.mnist_rotated_background_images``:  MNIST rotated background-images dataset module.
* ``datasets.mturk``:                            MTurk dataset module.
* ``datasets.mushrooms``:                        Mushrooms dataset module.
* ``datasets.newsgroups``:                       20-newsgroup dataset module.
* ``datasets.nips``:                             NIPS dataset module.
* ``datasets.occluded_faces_lfw``:               Labeled Faces in the Wild, occluded faces dataset module.
* ``datasets.occluded_mnist``:                   Occluded MNIST dataset module.
* ``datasets.ocr_letters``:                      OCR letters dataset module.
* ``datasets.rcv1``:                             RCV1 dataset module.
* ``datasets.rectangles``:                       Rectangles dataset module.
* ``datasets.rectangles_images``:                Rectangles images dataset module.
* ``datasets.sarcos``:                           SARCOS dataset module.
* ``datasets.scene``:                            Scene dataset module.
* ``datasets.web``:                              Web dataset module.
* ``datasets.yahoo_ltrc1``:                      Yahoo! Learning to Rank Challenge, Set 1 dataset module.
* ``datasets.yahoo_ltrc2``:                      Yahoo! Learning to Rank Challenge, Set 2 dataset module.
* ``datasets.yeast``:                            Yeast dataset module.
"""
