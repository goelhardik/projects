'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('binarized_mnist')
    except:
        print 'Could not download the dataset : ', 'binarized_mnist'
        assert False

def test_binarized_mnistloadToMemoryTrue():
    utGenerator.run_test('binarized_mnist', True)

def test_binarized_mnistloadToMemoryFalse():
    utGenerator.run_test('binarized_mnist', False)

def tearDown():
    dataset_store.delete('binarized_mnist')
