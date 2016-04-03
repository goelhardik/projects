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
Module ``datasets.letor_mq2007`` gives access to the LETOR 4.0 MQ2007
dataset, a learning to rank benchmark.

The LETOR 4.0 datasets are obtained here:
http://research.microsoft.com/en-us/um/beijing/projects/letor/letor4download.aspx.

**IMPORTANT:** the evaluation for this benchmark will require the use of
the official evaluation script, which can be downloaded at
http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Evaluation/Eval-Score-4.0.pl.txt.
Alternatively, function ``letor_evaluation`` in this module can be used.

| **Reference:** 
| LETOR 4.0 Datasets
| Microsoft Research
| http://research.microsoft.com/en-us/um/beijing/projects/letor/letor4dataset.aspx

"""

import mlpython.misc.io as mlio
import numpy as np
import os
from gzip import GzipFile as gfile

def load(dir_path,load_to_memory=False,fold=1):
    """
    Loads the LETOR 4.0 MQ2007 dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.

    This dataset comes with 5 predefined folds, which can be specified
    with option ``fold`` (default = 1). 
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'scores'``
    * ``'n_queries'``
    * ``'length'``

    """
    
    input_size=46
    dir_path = os.path.expanduser(dir_path)
    sparse=False

    if fold not in [1,2,3,4,5]:
        raise error('There are 5 predefined folds. Option fold should be an integer between 1 and 5')

    def convert(feature,value):
        if feature != 'qid':
            raise ValueError('Unexpected feature')
        return int(value)

    def load_line(line):
        return mlio.libsvm_load_line(line,convert,int,sparse,input_size)

    n_queries = [ [ 1017, 339, 336 ],
                  [ 1017, 336, 339 ],
                  [ 1014, 339, 339 ],
                  [ 1014, 339, 339 ],
                  [ 1014, 339, 339 ] ]

    lengths = [ [42158, 13813, 13652],
                [41958, 13652, 14013],
                [41320, 14013, 14290],
                [41478, 14290, 13855],
                [41955, 13855, 13813] ]
    
    # Get data file paths
    train_file,valid_file,test_file = [os.path.join(dir_path, 'MQ2007/Fold' + str(fold) + '/' + ds + '.txt') for ds in ['train','vali','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,),(1,)],[np.float64,int,int],l) for d,l in zip([train,valid,test],lengths[fold-1])]
        
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                        'scores':range(3),
                                        'n_queries':nq,
                                        'length':l,
                                        'n_pairs':l} for nq,l in zip(n_queries[fold-1],lengths[fold-1])]

    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}


def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """
    import os
    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset (this may take a little while)'
    import urllib,os
    urllib.urlretrieve('http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar',os.path.join(dir_path,'MQ2007.rar'))
    os.system('cd ' + dir_path +'; unrar x ' + os.path.join(dir_path,'MQ2007.rar'))
    print 'Done                     '


# Official evaluation script for LETOR 4.0 (in Perl)
perl_eval_script = r"""#!
# author: Jun Xu and Tie-Yan Liu
# modified by Jun Xu, March 3, 2009 (for Letor 4.0)
use strict;

#hash table for NDCG,
my %hsNdcgRelScore = (  "2", 3,
                        "1", 1,
                        "0", 0,
                    );

#hash table for Precision@N and MAP
my %hsPrecisionRel = ("2", 1,
                      "1", 1,
                      "0", 0
                );
#modified by Jun Xu, March 3, 2009
# for Letor 4.0. only output top 10 precision and ndcg
# my $iMaxPosition = 16;
my $iMaxPosition = 10;

my $argc = $#ARGV+1;
if($argc != 4)
{
		print "Invalid command line.\n";
		print "Usage: perl Eval.pl argv[1] argv[2] argv[3] argv[4]\n";
		print "argv[1]: feature file \n";
		print "argv[2]: prediction file\n";
		print "argv[3]: result (output) file\n";
		print "argv[4]: flag. If flag equals 1, output the evaluation results per query; if flag equals 0, simply output the average results.\n";
		exit -1;
}
my $fnFeature = $ARGV[0];
my $fnPrediction = $ARGV[1];
my $fnResult = $ARGV[2];
my $flag = $ARGV[3];
if($flag != 1 && $flag != 0)
{
	print "Invalid command line.\n";
	print "Usage: perl Eval.pl argv[1] argv[2] argv[3] argv[4]\n";
	print "Flag should be 0 or 1\n";
	exit -1;
}

my %hsQueryDocLabelScore = ReadInputFiles($fnFeature, $fnPrediction);
my %hsQueryEval = EvalQuery(\%hsQueryDocLabelScore);
OuputResults($fnResult, %hsQueryEval);


sub OuputResults
{
    my ($fnOut, %hsResult) = @_;
    open(FOUT, ">$fnOut");

    my @qids = sort{$a <=> $b} keys(%hsResult);
    my $numQuery = @qids;
    
#Precision@N and MAP
# modified by Jun Xu, March 3, 2009
# changing the output format
    print FOUT "qid\tP\@1\tP\@2\tP\@3\tP\@4\tP\@5\tP\@6\tP\@7\tP\@8\tP\@9\tP\@10\tMAP\n";
#---------------------------------------------
    my @prec;
    my $map = 0;
    for(my $i = 0; $i < $#qids + 1; $i ++)
    {
# modified by Jun Xu, March 3, 2009
# output the real query id    	
        my $qid = $qids[$i];
        my @pN = @{$hsResult{$qid}{"PatN"}};
        my $map_q = $hsResult{$qid}{"MAP"};
        if ($flag == 1)
        {
            print FOUT "$qid\t";
            for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
            {
                print FOUT sprintf("%.4f\t", $pN[$iPos]);
            }
            print FOUT sprintf("%.4f\n", $map_q);
        }
        for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
        {
            $prec[$iPos] += $pN[$iPos];
        }
        $map += $map_q;
    }
    print FOUT "Average\t";
    for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
    {
        $prec[$iPos] /= ($#qids + 1);
        print FOUT sprintf("%.4f\t", $prec[$iPos]);
    }
    $map /= ($#qids + 1);
    print FOUT sprintf("%.4f\n\n", $map);
    
#NDCG and MeanNDCG
# modified by Jun Xu, March 3, 2009
# changing the output format
    print FOUT "qid\tNDCG\@1\tNDCG\@2\tNDCG\@3\tNDCG\@4\tNDCG\@5\tNDCG\@6\tNDCG\@7\tNDCG\@8\tNDCG\@9\tNDCG\@10\tMeanNDCG\n";
#---------------------------------------------
    my @ndcg;
    my $meanNdcg = 0;
    for(my $i = 0; $i < $#qids + 1; $i ++)
    {
# modified by Jun Xu, March 3, 2009
# output the real query id
        my $qid = $qids[$i];
        my @ndcg_q = @{$hsResult{$qid}{"NDCG"}};
        my $meanNdcg_q = $hsResult{$qid}{"MeanNDCG"};
        if ($flag == 1)
        {
            print FOUT "$qid\t";
            for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
            {
                print FOUT sprintf("%.4f\t", $ndcg_q[$iPos]);
            }
            print FOUT sprintf("%.4f\n", $meanNdcg_q);
        }
        for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
        {
            $ndcg[$iPos] += $ndcg_q[$iPos];
        }
        $meanNdcg += $meanNdcg_q;
    }
    print FOUT "Average\t";
    for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
    {
        $ndcg[$iPos] /= ($#qids + 1);
        print FOUT sprintf("%.4f\t", $ndcg[$iPos]);
    }
    $meanNdcg /= ($#qids + 1);
    print FOUT sprintf("%.4f\n\n", $meanNdcg);

    close(FOUT);
}

sub EvalQuery
{
    my $pHash = $_[0];
    my %hsResults;
    
    my @qids = sort{$a <=> $b} keys(%$pHash);
    for(my $i = 0; $i < @qids; $i ++)
    {
        my $qid = $qids[$i];
        my @tmpDid = sort{$$pHash{$qid}{$a}{"lineNum"} <=> $$pHash{$qid}{$b}{"lineNum"}} keys(%{$$pHash{$qid}});
        my @docids = sort{$$pHash{$qid}{$b}{"pred"} <=> $$pHash{$qid}{$a}{"pred"}} @tmpDid;
        my @rates;

        for(my $iPos = 0; $iPos < $#docids + 1; $iPos ++)
        {
            $rates[$iPos] = $$pHash{$qid}{$docids[$iPos]}{"label"};
        }

        my $map  = MAP(@rates);
        my @PAtN = PrecisionAtN($iMaxPosition, @rates);
# modified by Jun Xu, calculate all possible positions' NDCG for MeanNDCG
        #my @Ndcg = NDCG($iMaxPosition, @rates);
        
        my @Ndcg = NDCG($#rates + 1, @rates);
        my $meanNdcg = 0;
        for(my $iPos = 0; $iPos < $#Ndcg + 1; $iPos ++)
        {
            $meanNdcg += $Ndcg[$iPos];
        }
        $meanNdcg /= ($#Ndcg + 1);
        
        
        @{$hsResults{$qid}{"PatN"}} = @PAtN;
        $hsResults{$qid}{"MAP"} = $map;
        @{$hsResults{$qid}{"NDCG"}} = @Ndcg;
        $hsResults{$qid}{"MeanNDCG"} = $meanNdcg;

    }
    return %hsResults;
}

sub ReadInputFiles
{
    my ($fnFeature, $fnPred) = @_;
    my %hsQueryDocLabelScore;
    
    if(!open(FIN_Feature, $fnFeature))
	{
		print "Invalid command line.\n";
		print "Open \$fnFeature\" failed.\n";
		exit -2;
	}
	if(!open(FIN_Pred, $fnPred))
	{
		print "Invalid command line.\n";
		print "Open \"$fnPred\" failed.\n";
		exit -2;
	}

    my $lineNum = 0;
    while(defined(my $lnFea = <FIN_Feature>))
    {
        $lineNum ++;
        chomp($lnFea);
        my $predScore = <FIN_Pred>;
        if (!defined($predScore))
        {
            print "Error to read $fnPred at line $lineNum.\n";
            exit -2;
        }
        chomp($predScore);
# modified by Jun Xu, 2008-9-9
# Labels may have more than 3 levels
# qid and docid may not be numeric
#        if ($lnFea =~ m/^([0-2]) qid\:(\d+).*?\#docid = (\d+)$/)

# modified by Jun Xu, March 3, 2009
# Letor 4.0's file format is different to Letor 3.0
#        if ($lnFea =~ m/^(\d+) qid\:([^\s]+).*?\#docid = ([^\s]+)$/)
        if ($lnFea =~ m/^(\d+) qid\:([^\s]+).*?\#docid = ([^\s]+) inc = ([^\s]+) prob = ([^\s]+).*$/)
        {
            my $label = $1;
            my $qid = $2;
            my $did = $3;
            my $inc = $4;
            my $prob= $5;
            $hsQueryDocLabelScore{$qid}{$did}{"label"} = $label;
            $hsQueryDocLabelScore{$qid}{$did}{"inc"} = $inc;
            $hsQueryDocLabelScore{$qid}{$did}{"prob"} = $prob;
            $hsQueryDocLabelScore{$qid}{$did}{"pred"} = $predScore;
            $hsQueryDocLabelScore{$qid}{$did}{"lineNum"} = $lineNum;
        }
        else
        {
            print "Error to parse $fnFeature at line $lineNum:\n$lnFea\n";
            exit -2;
        }
    }
    close(FIN_Feature);
    close(FIN_Pred);
    return %hsQueryDocLabelScore;
}


sub PrecisionAtN
{
    my ($topN, @rates) = @_;
    my @PrecN;
    my $numRelevant = 0;
#   modified by Jun Xu, 2009-4-24.
#   if # retrieved doc <  $topN, the P@N will consider the hole as irrelevant
#    for(my $iPos = 0;  $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
#
    for (my $iPos = 0; $iPos < $topN; $iPos ++)
    {
        my $r;
        if ($iPos < $#rates + 1)
        {
            $r = $rates[$iPos];
        }
        else
        {
            $r = 0;
        }
        $numRelevant ++ if ($hsPrecisionRel{$r} == 1);
        $PrecN[$iPos] = $numRelevant / ($iPos + 1);
    }
    return @PrecN;
}

sub MAP
{
    my @rates = @_;

    my $numRelevant = 0;
    my $avgPrecision = 0.0;
    for(my $iPos = 0; $iPos < $#rates + 1; $iPos ++)
    {
        if ($hsPrecisionRel{$rates[$iPos]} == 1)
        {
            $numRelevant ++;
            $avgPrecision += ($numRelevant / ($iPos + 1));
        }
    }
    return 0.0 if ($numRelevant == 0);
    #return sprintf("%.4f", $avgPrecision / $numRelevant);
    return $avgPrecision / $numRelevant;
}

sub DCG
{
    my ($topN, @rates) = @_;
    my @dcg;
    
    $dcg[0] = $hsNdcgRelScore{$rates[0]};
#   Modified by Jun Xu, 2009-4-24
#   if # retrieved doc <  $topN, the NDCG@N will consider the hole as irrelevant
#    for(my $iPos = 1; $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
#
    for(my $iPos = 1; $iPos < $topN; $iPos ++)
    {
        my $r;
        if ($iPos < $#rates + 1)
        {
            $r = $rates[$iPos];
        }
        else
        {
            $r = 0;
        }
        if ($iPos < 2)
        {
            $dcg[$iPos] = $dcg[$iPos - 1] + $hsNdcgRelScore{$r};
        }
        else
        {
            $dcg[$iPos] = $dcg[$iPos - 1] + ($hsNdcgRelScore{$r} * log(2.0) / log($iPos + 1.0));
        }
    }
    return @dcg;
}
sub NDCG
{
    my ($topN, @rates) = @_;
    my @ndcg;
    my @dcg = DCG($topN, @rates);
    my @stRates = sort {$hsNdcgRelScore{$b} <=> $hsNdcgRelScore{$a}} @rates;
    my @bestDcg = DCG($topN, @stRates);
    
    for(my $iPos =0; $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
    {
        $ndcg[$iPos] = 0;
        $ndcg[$iPos] = $dcg[$iPos] / $bestDcg[$iPos] if ($bestDcg[$iPos] != 0);
    }
    return @ndcg;
}
"""

def letor_evaluation(outputs,evaluation_set,fold=1,dir_path=None):
    """
    Returns the lists of precisions and NDCG performance measures,
    based on some given ``outputs`` and the ``evaluation_set`` 
    (``'train'``,``'valid'`` or ``'test'``).

    Precisions and NDCG are measured at cutoffs from 1 to 10. The last
    element of each list is the mean average precision (MAP) and mean
    NDCG.
    """

    import os,time,random
    eval_set_to_file = {'train':'train.txt','valid':'vali.txt','test':'test.txt'}
    if evaluation_set not in eval_set_to_file:
        raise error('evaluation_set should be one of the following: \'train\',\'valid\',\'test\'')
    if dir_path is None:
        repo = os.environ.get('MLPYTHON_DATASET_REPO')
        ground_truth = os.path.join(repo,'letor_mq2007/MQ2007/Fold' + str(fold) + '/' + eval_set_to_file[evaluation_set])
    else:
        ground_truth = os.path.join(dir_path,'MQ2007/Fold' + str(fold) + '/' + eval_set_to_file[evaluation_set])


    def read_output(out_file):
        f = open(out_file)
        lines = f.readlines()
        
        prec = [ float(t) for t in lines[1].split('\t')[1:] ]
        ndcg = [ float(t) for t in lines[4].split('\t')[1:] ]
        return (prec,ndcg)

    # Use official script to get errors
    tmp_file_pred = str(time.clock()) + str(random.random()) + '_pred.txt'
    tmp_file_out = str(time.clock()) + str(random.random()) + '_out.txt'
    tmp_file_eval = str(time.clock()) + str(random.random()) + '_eval.pl'
    
    # Create tmp_file with predicted score based on outputs
    f = open(tmp_file_pred,'w')
    for o in outputs:
        for o_i in o:
            f.write(str(-o_i)+'\n')
    f.close()
    
    # Write evaluation script
    f = open(tmp_file_eval,'w')
    f.write(perl_eval_script)
    f.close()
    
    os.system('perl ' + tmp_file_eval + ' ' + ground_truth + ' ' + tmp_file_pred + ' ' + tmp_file_out + ' 0' )
    os.system('rm -f ' + tmp_file_pred)
    prec,ndcg = read_output(tmp_file_out)
    os.system('rm -f ' + tmp_file_out)
    os.system('rm -f ' + tmp_file_eval)
    
    return prec,ndcg
