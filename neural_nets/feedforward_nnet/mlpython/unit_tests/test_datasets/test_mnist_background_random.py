'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mnist_background_random')
    except:
        print 'Could not download the dataset : ', 'mnist_background_random'
        assert False

def test_mnist_background_randomloadToMemoryTrue():
    utGenerator.run_test('mnist_background_random', True)

def test_mnist_background_randomloadToMemoryFalse():
    utGenerator.run_test('mnist_background_random', False)

def tearDown():
    dataset_store.delete('mnist_background_random')
