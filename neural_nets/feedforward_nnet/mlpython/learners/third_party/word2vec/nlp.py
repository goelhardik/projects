# Copyright 2014 Frederic Bergeron & Benoit Gauthier. All rights reserved.
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

import subprocess
import uuid
import os
import tempfile
import shutil
from mlpython.learners.generic import Learner
import numpy as np



class Word2Vec(Learner):
    """
    Word representation learning with word2vec package.

    Option ``output_file_name`` is the file to save the resulting word vectors / word clusters.

    Option ``size`` is the size of the word vectors; default is 100.

    Option ``window`` is the max skip length between words; default is 5.

    Option ``sample`` is the threshold for occurrence of words. Those that appear with higher frequency
    in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5.

    Option ``use_historical_softmax`` is whether to use the Hierarchical Softmax; default is 1 (0 = not used).

    Option ``negative`` is the number of negative examples; default is 0, common values are 5 - 10 (0 = not used).

    Option ``number_of_threads`` is the self-explanatory (default 1).

    Option ``minimum_word_count`` is whether to discard words that appear less than <int> times; default is 5.

    Option ``alpha`` is the starting learning rate; default is 0.025.

    Option ``use_classes`` is whether to output word classes rather than word vectors; default number of classes is 0 (vectors are written)

    Option ``save_vector_as_binary`` is whetehr to save the resulting vectors in binary mode; default is 0 (off).

    Option ``use_continuous_bag_of_words`` is whether to use the continuous bag of words model; default is 0 (skip-gram model).
    """

    def __init__(self, 
                 delete_temporary_files = True,
                 output_file_name = "word2vec_representation",
                 size = 100,
                 window =5,
                 sample = 0,
                 use_historical_softmax =1,
                 negative = 0,
                 number_of_threads = 1,
                 minimum_word_count = 5,
                 alpha = 0.025,
                 use_classes = 0,
                 save_vector_as_binary = 0,
                 use_continuous_bag_of_words = 0,
                 display_script_output = False,
                 temporary_dir = None):
        self.is_trained = False
        self.delete_created_files = delete_temporary_files
        self.output_file_name = output_file_name
        self.use_continuous_bag_of_words = use_continuous_bag_of_words
        self.size = size
        self.window = window
        self.sample = sample
        self.use_historical_softmax = use_historical_softmax
        self.negative = negative
        self.number_of_threads = number_of_threads
        self.minimum_word_count = minimum_word_count
        self.alpha = alpha
        self.use_classes = use_classes
        self.save_vector_as_binary = save_vector_as_binary
        self.temporary_folder = None
        
        self.script_output = None
        self.display_script_output = display_script_output 
        #"vectors.bin -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1 ./distance vectors.bin"

        # Creating temporary directory for word2vec
        if temporary_dir is None :
            unique_id = uuid.uuid4().hex
            self.temporary_folder = tempfile.gettempdir()+'/word2vectmp_'+unique_id
        else:
            unique_id = uuid.uuid4().hex
            self.temporary_folder = temporary_dir+'/word2vectmp_'+unique_id

        if os.path.exists(self.temporary_folder) :
            raise ValueError('Temporary directory ' + self.temporary_folder + ' already exists')
        os.makedirs(self.temporary_folder)

    def train(self,trainset):
        
        if self.is_trained :
            return

        self.script_output = None
        if not self.display_script_output:
            self.script_output = open(os.devnull,'w')
       
        input_name = os.path.join(self.temporary_folder,'inputword2vec'+'.tmp')
        self.output_name = os.path.join(self.temporary_folder, self.output_file_name)

        input_file = open(input_name,'w')
        tmplist = list()

        '''iterate through the files'''

        for x in trainset:
            tmplist.append(' '.join(x))
        input_file.write(' '.join(tmplist))
        input_file.close()

        # Get the location of the shell script
        process_path = 'word2vec'
        args =list()
        args.append(process_path)
        args.append('-train')
        args.append(input_name)
        args.append('-output')
        args.append(self.output_name)
        args.append('-cbow')
        args.append(str(self.use_continuous_bag_of_words))
        args.append('-size')
        args.append(str(self.size))
        args.append('-window')
        args.append(str(self.window))
        args.append('-negative')
        args.append(str(self.negative))
        args.append('-sample')
        args.append(process_path)
        args.append('-threads')
        args.append(str(self.number_of_threads))
        args.append('-binary')
        args.append(str(self.save_vector_as_binary))

        sub = subprocess.Popen(args)
        sub.wait()
        print '\n' #because the subprocess doesn't end its last line


    def use(self, dataset):
        """Returns a list of numpy matrixes containing vector retresentation of each word in each file of the dataset.
        Out Of Vocabulary words are represented by a zeros array"""

        trained_dict = self.get_word_representations()
        with open(self.output_name, 'r') as f:
            first_line = f.readline()
        nb_of_words, vector_lenght = first_line.split()

        used_list = []
        for file in dataset:
            used = np.zeros((len(file),int(vector_lenght)))
            for index, word in enumerate(file):
                try:
                    used[index] = trained_dict[word]
                except KeyError:
                    pass #since there is no OOF word
        used_list.append(used)

        return used_list

    def forget(self):
        """Remove the tree of the temporary file."""
        shutil.rmtree(self.temporary_folder)
        return

    def test(self, dataset):
        raise NotImplementedError

    def get_word_representations(self):
        """Get vector reprensentation for each word in the training corpus

        Returns a dictionnary in the format of ``word => representation``"""

        trained_dict = {}
        with open(self.output_name) as f:
            for line in f:
                line = line.strip().split(' ')
                trained_dict[line[0]] = np.array(line[1:], dtype=np.float64)

        return trained_dict

    def __del__(self):
        if self.delete_created_files:
            self.forget()
