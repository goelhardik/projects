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
The ``learners.ranking`` module contains learners meant for ranking problems. 
The MLProblems for these learners should be iterators over 
triplets (input,target,query), where input is a list of
document representations and target is a list of associated 
relevance scores for the given query.

The currently implemented algorithms are:

* RankingFromClassifier:  a ranking model based on a classifier.
* RankingFromRegression:  a ranking model based on a regression model.
* ListNet:                ListNet ranking model.

"""

from generic import Learner,OnlineLearner
import numpy as np
import mlpython.mlproblems.ranking as mlpb

def default_merge(input, query):
    return input

def err_and_ndcg(output,target,max_score,k=10):
    """
    Computes the ERR and NDCG score 
    (taken from here: http://learningtorankchallenge.yahoo.com/evaluate.py.txt)
    """

    err = 0.
    ndcg = 0.
    l = [int(x) for x in target]
    r = [int(x)+1 for x in output]
    nd = len(target) # Number of documents
    assert len(output)==nd, 'Expected %d ranks, but got %d.'%(nd,len(r))
    
    gains = [-1]*nd # The first element is the gain of the first document in the predicted ranking
    assert max(r)<=nd, 'Ranks larger than number of documents (%d).'%(nd)
    for j in range(nd):
      gains[r[j]-1] = (2.**l[j]-1.0)/(2.**max_score)
    assert min(gains)>=0, 'Not all ranks present.'
    
    p = 1.0
    for j in range(nd):
        r = gains[j]
        err += p*r/(j+1.0)
        p *= 1-r
    
    dcg = sum([g/np.log(j+2) for (j,g) in enumerate(gains[:k])])
    gains.sort()
    gains = gains[::-1]
    ideal_dcg = sum([g/np.log(j+2) for (j,g) in enumerate(gains[:k])])
    if ideal_dcg:
        ndcg += dcg / ideal_dcg
    else:
        ndcg += 0.5
        
    return (err,ndcg)


class RankingFromClassifier(Learner):
    """ 
    A ranking model based on a classifier.
 
    This learner trains a given classifier to 
    predict the target relevance score associated to each
    document/query pairs found in the training set.
    
    Option ``classifier`` is the classifier to train.

    The classifier can be used for ranking based on three
    measures, specified by option ``ranking_measure``: 

    * ``ranking_measure='predicted_score':``
      the predicted relevance score is used (first output 
      of classifier);
    * ``ranking_measure='expected_score':``
      the distribution over scores (second output) is
      used to computed the expected score, and a ranking
      is determined by sorting those expectations;
    * ``ranking_measure='expected_persistence':``
      the distribution over scores is used to determine
      the expected persistence (``(2**score-1)/max_score``).
      Ranking according to this measure should work well
      for the ERR ranking error.

    To use ``ranking_measure='predicted_score'`` as the ranking
    measure, the classifier can have only one output, i.e. the
    predicted score.  To use the other two ranking measures, the
    classifier must also output a distribution over possible relevance
    scores as the second output.

    Option ``merge_document_and_query`` should be a 
    callable function that takes two arguments (the 
    input document and the query) and outputs a 
    merged representation for the pair which will
    be fed to the classifier. By default, it is assumed
    that the document representation already contains
    query information, and only the document the input
    document is returned.

    **Required metadata:**
    
    * ``'scores'``

    """
    def __init__(   self,
                    classifier,
                    merge_document_and_query = default_merge,
                    ranking_measure = 'expected_score',
                    ):
        self.stage = 0
        self.classifier = classifier
        self.merge_document_and_query=merge_document_and_query
        self.ranking_measure = ranking_measure
        if ranking_measure not in set(['expected_score','expected_persistence','predicted_score']):
            raise ValueError, 'Invalid ranking measure \'%s\''%ranking_measure

    def train(self,trainset):
        """
        Trains the classifier on the merged documents and queries.
        Each call to train increments self.stage by 1.
        """

        classifier_trainset = mlpb.RankingToClassificationProblem(trainset,
                                                                  trainset.metadata,
                                                                  merge_document_and_query = self.merge_document_and_query)
        self.classifier_trainset_metadata = classifier_trainset.metadata
        self.max_score = max(trainset.metadata['scores'])
        
        # Training classifier
        self.classifier.train(classifier_trainset)

        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.classifier.forget()
        #self.classifier_trainset=None
        self.classifier_trainset_metadata=None

    def use(self,dataset):
        """
        Outputs a list corresponding to the position (starting at 0) of each
        document corresponding to its relevance score (from most relevant to least). 

        For example, ordering ``[1,3,0,2]`` means that the 
        first document is the second most relevant, the second document
        is the fourth most relevant, the third document is the first most
        relevant and the fourth document is the third most relevant.

        Inspired from http://learningtorankchallenge.yahoo.com/instructions.php
        """
        
        cdataset = mlpb.RankingToClassificationProblem(dataset,dataset.metadata,call_setup=False,
                                                       merge_document_and_query = self.merge_document_and_query)
        cdataset.metadata['class_to_id'] = self.classifier_trainset_metadata['class_to_id']
        cdataset.metadata['targets'] = self.classifier_trainset_metadata['targets']
        cdataset.class_to_id = cdataset.metadata['class_to_id']

        coutputs = self.classifier.use(cdataset)
        offset = 0
        outputs = []

        if self.ranking_measure == 'expected_score' or self.ranking_measure == 'expected_persistence':
            # Create vector of measures appropriate for computing the necessary expectations, 
            # ordered according to the class ID mapping:
            score_to_class_id = self.classifier_trainset_metadata['class_to_id']
            ordered_measures = np.zeros((len(score_to_class_id)))
            for k,v in score_to_class_id.iteritems():
                if self.ranking_measure == 'expected_score':
                    ordered_measures[v] = k
                elif self.ranking_measure == 'expected_persistence':
                    ordered_measures[v] = (2.**k-1.0)/(2.**self.max_score)

        for inputs,targets,query in dataset:
            if self.ranking_measure == 'predicted_score':
                preds = [ -co[0] for co in coutputs[offset:(offset+len(inputs))]]
            elif self.ranking_measure == 'expected_score' or self.ranking_measure == 'expected_persistence':
                preds = [ -np.dot(ordered_measures,co[1]) for co in coutputs[offset:(offset+len(inputs))]]
            ordered = np.argsort(preds)
            order = np.zeros(len(ordered))
            order[ordered] = range(len(ordered))
            outputs += [order]
            offset += len(inputs)
        return outputs

    def test(self,dataset):
        """
        Outputs the document ordering and the associated ERR and NDCG scores.
        """
        outputs = self.use(dataset)
        assert len(outputs) == len(dataset)
        costs = np.zeros((len(dataset),2))
        for output,cost,example in zip(outputs,costs,dataset):
            cost[0],cost[1] = err_and_ndcg(output,example[1],self.max_score)

        return outputs,costs

class RankingFromRegression(Learner):
    """ 
    A ranking model based on a regression model.
 
    This learner trains a given regression learner to 
    predict the target relevance score associated to each
    document/query pairs found in the training set.

    Option ``regression`` is the regression model to train.

    Option ``merge_document_and_query`` should be a 
    callable function that takes two arguments (the 
    input document and the query) and outputs a 
    merged representation for the pair which will
    be fed to the regression model. By default, it is assumed
    that the document representation already contains
    query information, and only the document the input
    document is returned.

    **Required metadata:**
    
    * ``'scores'``

    """
    def __init__(   self,
                    regression,
                    merge_document_and_query = default_merge):
        self.stage = 0
        self.regression = regression
        self.merge_document_and_query=merge_document_and_query

    def train(self,trainset):
        """
        Trains the regression model on the merged documents and queries.
        Each call to train increments self.stage by 1.
        """

        regression_trainset = mlpb.RankingToRegressionProblem(trainset,
                                                              trainset.metadata,
                                                              merge_document_and_query = self.merge_document_and_query)
        self.regression_trainset_metadata = regression_trainset.metadata
        self.max_score = max(trainset.metadata['scores'])
        
        # Training classifier
        self.regression.train(regression_trainset)

        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.regression.forget()
        self.regression_trainset_metadata=None

    def use(self,dataset):
        """
        Outputs a list corresponding to the position (starting at 0) of each
        document corresponding to its relevance score (from most relevant to least). 

        For example, ordering ``[1,3,0,2]`` means that the 
        first document is the second most relevant, the second document
        is the fourth most relevant, the third document is the first most
        relevant and the fourth document is the third most relevant.

        Inspired from http://learningtorankchallenge.yahoo.com/instructions.php
        """
        
        cdataset = mlpb.RankingToRegressionProblem(dataset,dataset.metadata,call_setup=False)
        cdataset.merge_document_and_query = self.merge_document_and_query

        coutputs = self.regression.use(cdataset)
        offset = 0
        outputs = []
        for inputs,targets,query in dataset:
            preds = [ -co[0] for co in coutputs[offset:(offset+len(inputs))]]
            ordered = np.argsort(preds)
            order = np.zeros(len(ordered))
            order[ordered] = range(len(ordered))
            outputs += [order]
            offset += len(inputs)
        return outputs

    def test(self,dataset):
        """
        Outputs the document ordering and the associated ERR and NDCG scores.
        """
        outputs = self.use(dataset)
        assert len(outputs) == len(dataset)
        costs = np.zeros((len(dataset),2))
        for output,cost,example in zip(outputs,costs,dataset):
            cost[0],cost[1] = err_and_ndcg(output,example[1],self.max_score)

        return outputs,costs

class ListNet(OnlineLearner):
    """ 
    ListNet ranking model.
 
    This implementation only models the distribution of documents
    appearing first in the ranked list (this is the setting favored in
    the experiments of the original ListNet paper). ListNet is trained
    by minimizing the KL divergence between a target distribution
    derived from the document scores and ListNet's output
    distribution.

    Option ``n_stages`` is the number of training iterations over the
    training set.

    Option ``hidden_size`` determines the size of the hidden layer (default = 50).

    Option ``learning_rate`` is the learning rate for stochastic
    gradient descent training (default = 0.01).

    Option ``weight_per_query`` determines whether to weight each
    ranking example (one for each query) by the number of documents to
    rank. If True, the effect is to multiply the learning rate by
    the number of documents for the current query. If False, no weighting
    is applied (default = False).

    Option ``alpha`` controls the entropy of the target distribution
    ListNet is trying to predict: ``target = exp(alpha *
    scores)/sum(exp(alpha * scores))`` (default = 1.).

    Option ``merge_document_and_query`` should be a 
    callable function that takes two arguments (the 
    input document and the query) and outputs a 
    merged representation for the pair which will
    be fed to ListNet. By default, it is assumed
    that the document representation already contains
    query information, and only the document the input
    document is returned.

    Option ``seed`` determines the seed of the random number generator
    used to initialize the model.

    **Required metadata:**
    
    * ``'scores'``

    | **Reference:** 
    | Learning to Rank: From Pairwise Approach to Listwise Approach
    | Cao, Qin, Liu, Tsai and Li
    | http://research.microsoft.com/pubs/70428/tr-2007-40.pdf

    """

    def __init__(self, n_stages, hidden_size = 50,
                 learning_rate = 0.01,
                 weight_per_query = False,
                 alpha = 1.,
                 merge_document_and_query = default_merge,
                 seed = 1234):

        self.n_stages = n_stages
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_per_query = weight_per_query
        self.alpha = alpha
        self.merge_document_and_query = merge_document_and_query
        self.seed = seed
        
        self.stage = 0

    def initialize_learner(self,metadata):
      self.rng = np.random.mtrand.RandomState(self.seed)
      input_size = metadata['input_size']
      self.max_score = max(metadata['scores'])
      self.V = (2*self.rng.rand(input_size,self.hidden_size)-1)/input_size

      self.c = np.zeros((self.hidden_size))
      self.W = (2*self.rng.rand(self.hidden_size,1)-1)/self.hidden_size
      self.b = np.zeros((1))
         

    def update_learner(self,example):
      input_list = example[0]
      relevances = example[1] 
      query = example[2]
      n_documents = len(input_list)

      target_probs = np.zeros((n_documents,1))
      input_size = input_list[0].shape[0]
      inputs = np.zeros((n_documents,input_size))

      for t,r,il,input in zip(target_probs,relevances,input_list,inputs):
         t[0] = np.exp(self.alpha*r)
         input[:input_size] = self.merge_document_and_query(il,query)
      target_probs = target_probs/np.sum(target_probs,axis=0)

      hid = np.tanh(np.dot(inputs,self.V)+self.c)

      outact = np.dot(hid,self.W) + self.b
      outact -= np.max(outact)
      expout = np.exp(outact)
      output = expout/np.sum(expout,axis=0)

      doutput = output-target_probs
      dhid = np.dot(doutput,self.W.T)*(1-hid**2)

      if self.weight_per_query:
         lr = self.learning_rate*n_documents
      else:
         lr = self.learning_rate
      self.W -= lr * np.dot(hid.T,doutput)
      self.b -= lr * np.sum(doutput)
      self.V -= lr * np.dot(inputs.T,dhid)
      self.c -= lr * np.sum(dhid,axis=0)

    def use_learner(self,example):
      input_list = example[0]
      n_documents = len(input_list)
      query = example[2]

      input_size = input_list[0].shape[0]
      inputs = np.zeros((n_documents,input_size))
      for il,input in zip(input_list,inputs):
         input[:input_size] = self.merge_document_and_query(il,query)
      
      hid = np.tanh(np.dot(inputs,self.V)+self.c)
      outact = np.dot(hid,self.W) + self.b
      outact -= np.max(outact)
      expout = np.exp(outact)
      output = expout/np.sum(expout,axis=0)
      ordered = np.argsort(-output.ravel())
      order = np.zeros(len(ordered))
      order[ordered] = range(len(ordered))
      return order

    def cost(self,output,example):
      return err_and_ndcg(output,example[1],self.max_score)
