'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mturk')
    except:
        print 'Could not download the dataset : ', 'mturk'
        assert False

def test_mturkloadToMemoryTrue():
    utGenerator.run_test('mturk', True)

def test_mturkloadToMemoryFalse():
    utGenerator.run_test('mturk', False)

def tearDown():
    dataset_store.delete('mturk')
