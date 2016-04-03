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
The ``datasets.store`` module provides a unique interface for downloading datasets
and creating MLProblems from those datasets.

It defines the following variables:

* ``datasets.store.all_names``:             set of all dataset names
* ``datasets.store.classification_names``:  set of dataset names for classification
* ``datasets.store.regression_names``:      set of dataset names for regression
* ``datasets.store.distribution_names``:    set of dataset names for distribution estimation
* ``datasets.store.multilabel_names``:      set of dataset names for multilabel classification
* ``datasets.store.multiregression_names``: set of dataset names for multidimensional regression
* ``datasets.store.ranking_names``:         set of dataset names for ranking problems

It also defines the following functions:

* ``datasets.store.download``:                    downloads a given dataset
* ``datasets.store.get_classification_problem``:  returns train/valid/test classification MLProblems from some given dataset name
* ``datasets.store.get_regression_problem``:      returns train/valid/test regression MLProblems from some given dataset name
* ``datasets.store.get_distribution_problem``:    returns train/valid/test distribution estimation MLProblems from some given dataset name
* ``datasets.store.get_multilabel_problem``:      returns train/valid/test multilabel classification MLProblems from some given dataset name
* ``datasets.store.get_multiregression_problem``: returns train/valid/test multidimensional regression MLProblems from some given dataset name
* ``datasets.store.get_ranking_problem``:         returns train/valid/test ranking MLProblems from some given dataset name
* ``datasets.store.get_k_fold_experiment``:       returns a list of train/valid/test MLProblems for a k-fold experiment
* ``get_semisupervised_experiment``:              returns new train/valid/test MLProblems corresponding to a semi-supervised learning experiment

"""

classification_names = set(['adult',
                            'connect4',
                            'convex',
                            'dna',
                            'heart',
                            'mnist',
                            'mnist_basic',
                            'mnist_background_images',
                            'mnist_background_random',
                            'mnist_rotated',
                            'mnist_rotated_background_images',
                            'mushrooms',
                            'newsgroups',
                            'ocr_letters',
                            'rcv1',
                            'rectangles',
                            'rectangles_images',
                            'web'])
                            
regression_names = set(['abalone',
                        'cadata',
                        'housing'])

distribution_names = set(['adult',
                    'binarized_mnist',
                    'connect4',
                    'dna',
                    'heart',
                    'mnist',
                    'mushrooms',
                    'nips',
                    'ocr_letters',
                    'rcv1',
                    'web'])

multilabel_names = set(['bibtex',
                        'corel5k',
                        'corrupted_ocr_letters',
                        'corrupted_mnist',
                        'majmin',
                        'mediamill',
                        'medical',
                        'mturk',
                        'occluded_mnist',
                        'scene',
                        'yeast'])

multiregression_names = set(['occluded_faces_lfw',
                             'face_completion_lfw',
                             'sarcos'])

ranking_names = set(['yahoo_ltrc1',
                     'yahoo_ltrc2',
                     'letor_mq2007',
                     'letor_mq2008'])

#nlp_names = set(['reuters_v1'])

all_names = distribution_names | classification_names | multilabel_names | multiregression_names | regression_names | ranking_names #| nlp_names

def download(name,dataset_dir=None):
    """
    Downloads dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``all_names`` of this module).

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, a subdirectory will be created and the
    dataset will be downloaded there. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in all_names:
        raise ValueError('dataset '+name+' unknown')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'
    import os
    if dataset_dir is None:
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    mldataset.obtain(dataset_dir)

def delete(name):
    """Remove the dataset from the hard drive"""
    import os
    import shutil
    repo = os.environ.get('MLPYTHON_DATASET_REPO')
    if repo is None:
        raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
    dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    if not os.path.exists(dataset_dir):
        raise ValueError('The directory '+ repo +' does not exists')
    shutil.rmtree(dataset_dir)
    
def get_classification_problem(name,dataset_dir=None,load_to_memory=True,**kw):
    """
    Creates train/valid/test classification MLProblems from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``classification_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in classification_names:
        raise ValueError('dataset '+name+' unknown for classification learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory,**kw)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.classification as mlpb
    trainset = mlpb.ClassificationProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_regression_problem(name,dataset_dir=None,load_to_memory=True,**kw):
    """
    Creates train/valid/test regression MLProblems from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``regression_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in regression_names:
        raise ValueError('dataset '+name+' unknown for regression learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory,**kw)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.generic as mlpb
    trainset = mlpb.MLProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_distribution_problem(name,dataset_dir=None,load_to_memory=True,**kw):
    """
    Creates train/valid/test distribution estimation MLProblems from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``distribution_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in distribution_names:
        raise ValueError('dataset '+name+' unknown for distribution learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory,**kw)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.generic as mlpb
    if name == 'binarized_mnist' or name == 'nips': 
        trainset = mlpb.MLProblem(train_data,train_metadata)
    else:
        trainset = mlpb.SubsetFieldsProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_multilabel_problem(name,dataset_dir=None,load_to_memory=True,**kw):
    """
    Creates train/valid/test multilabel classification MLProblems from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``multilabel_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in multilabel_names:
        raise ValueError('dataset '+name+' unknown for multi-label classification learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory,**kw)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.generic as mlpb
    trainset = mlpb.MLProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_multiregression_problem(name,dataset_dir=None,load_to_memory=True,**kw):
    """
    Creates train/valid/test multidimensional regression MLProblems from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``multiregression_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in multiregression_names:
        raise ValueError('dataset '+name+' unknown for multidimensional regression learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory,**kw)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.generic as mlpb
    trainset = mlpb.MLProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_ranking_problem(name,dataset_dir=None,load_to_memory=True,**kw):
    """
    Creates train/valid/test ranking MLProblems from dataset ``name``.

    ``name`` must be one of the supported dataset (see variable
    ``ranking_names`` of this module).

    Option ``load_to_memory`` determines whether the dataset should
    be loaded into memory or always read from its files.

    If environment variable MLPYTHON_DATASET_REPO has been set to a
    valid directory path, this function will look into its appropriate
    subdirectory to find the dataset. Alternatively the subdirectory path
    can be given by the user through option ``dataset_dir``.
    """

    if name not in ranking_names:
        raise ValueError('dataset '+name+' unknown for ranking learning')
    
    exec 'import mlpython.datasets.'+name+' as mldataset'

    if dataset_dir is None:
        # Try to find dataset in MLPYTHON_DATASET_REPO
        import os
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        if repo is None:
            raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
        dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name

    all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory,**kw)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    test_data, test_metadata = all_data['test']

    import mlpython.mlproblems.ranking as mlpb
    trainset = mlpb.RankingProblem(train_data,train_metadata)
    validset = trainset.apply_on(valid_data,valid_metadata)
    testset = trainset.apply_on(test_data,test_metadata)

    return trainset,validset,testset

def get_k_fold_experiment(datasets,k=10,seed=1234):
    """
    Creates a k-fold experiment from a list of MLProblems ``datasets``.

    ``k`` determines the number of folds, and ``seed`` is for the
    random number generator that will shuffle all the examples before
    creating the folds.

    The output is a list of ``k`` triplets ``(train,valid,test)``, which
    determine the experiment to be run for each ``test`` fold. ``valid``
    is also an individual fold and ``train`` corresponds to the concatenation
    of the remaining folds.

    """

    import mlpython.mlproblems.generic as mlpb
    import numpy as np

    all_data = mlpb.MergedProblem(datasets)

    # Shuffle data ids
    ids = range(len(all_data))
    rng = np.random.mtrand.RandomState(seed)
    rng.shuffle(ids)

    # Create folds
    fold_size = int(np.floor(float(len(all_data))/k))
    fold_ids = []
    beg = 0
    for f in range(k-1):
        fold_ids += [ids[beg:(beg+fold_size)]]
        beg += fold_size
    # Put rest of data in last fold
    fold_ids += [ids[beg:]]
    folds = [ mlpb.SubsetProblem(all_data,subset=set(f_ids)) for f_ids in fold_ids ]

    # Create each fold's experiment
    k_fold_experiment = []
    for f in range(k):
        train_folds = folds[:f] + folds[(f+1):]
        test = folds[f]
        valid = train_folds[-1]
        train_folds = train_folds[:-1]
        train = mlpb.MergedProblem(train_folds)
        k_fold_experiment += [(train,valid,test)]

    return k_fold_experiment



def get_semisupervised_experiment(trainset,validset,testset,labeled_frac=0.1,label_field=1,seed=1234):
    """
    Creates a semi-supervised experiment from training, validation and
    test MLProblems.

    The test set is returned untouched. The training and validation
    sets are regenerated so that the ratio of validation/training
    labeled data size is the same as in the original datasets.

    ``labeled_frac`` is the total fraction of labeled data in the
    training and validation sets. Only the training set will contain
    unlabeled data.

    ``label_field`` is the index for the examples' label field.

    ``seed`` is for the random number generator that will select which
    examples to keep labeled and which to put in the validation set.

    """

    import mlpython.mlproblems.generic as mlpb
    import numpy as np

    train_valid_data = mlpb.MergedProblem([trainset,validset])

    # Shuffle data ids to make new train/valid split
    ids = range(len(train_valid_data))
    rng = np.random.mtrand.RandomState(seed)
    rng.shuffle(ids)

    # Figure out number of labeled/unlabeled examples
    n_total_labeled = int(labeled_frac*float(len(train_valid_data)))
    n_total_unlabeled = len(train_valid_data)-n_total_labeled

    # Figure out train/valid split ratio from original data
    train_frac = float(len(trainset))/len(train_valid_data)
    n_train_labeled = int(train_frac*float(n_total_labeled))
    n_valid_labeled = n_total_labeled - n_train_labeled

    # Make train/valid split
    new_trainset = mlpb.SubsetProblem(train_valid_data,subset=set(ids[:(n_train_labeled+n_total_unlabeled)]))
    new_validset = mlpb.SubsetProblem(train_valid_data,subset=set(ids[(n_train_labeled+n_total_unlabeled):]))
    if len(new_validset) != n_valid_labeled:
        raise ValueError('Something is wrong!')

    # Unlabel some of the training examples
    unlabeled_ids = range(len(new_trainset))
    rng.shuffle(unlabeled_ids)
    unlabeled_ids = unlabeled_ids[:n_total_unlabeled]
    new_trainset = mlpb.SemisupervisedProblem(new_trainset,unlabeled_ids = unlabeled_ids,label_field=label_field)

    return new_trainset,new_validset,testset
