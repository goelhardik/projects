'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('corrupted_mnist')
    except:
        print 'Could not download the dataset : ', 'corrupted_mnist'
        assert False

def test_corrupted_mnistloadToMemoryTrue():
    utGenerator.run_test('corrupted_mnist', True)

def test_corrupted_mnistloadToMemoryFalse():
    utGenerator.run_test('corrupted_mnist', False)

def tearDown():
    dataset_store.delete('corrupted_mnist')
