'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mnist_basic')
    except:
        print 'Could not download the dataset : ', 'mnist_basic'
        assert False

def test_mnist_basicloadToMemoryTrue():
    utGenerator.run_test('mnist_basic', True)

def test_mnist_basicloadToMemoryFalse():
    utGenerator.run_test('mnist_basic', False)

def tearDown():
    dataset_store.delete('mnist_basic')
