'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('nips')
    except:
        print 'Could not download the dataset : ', 'nips'
        assert False

def test_nipsloadToMemoryTrue():
    utGenerator.run_test('nips', True)

def test_nipsloadToMemoryFalse():
    utGenerator.run_test('nips', False)

def tearDown():
    dataset_store.delete('nips')
