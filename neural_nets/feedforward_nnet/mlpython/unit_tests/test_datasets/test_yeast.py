'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('yeast')
    except:
        print 'Could not download the dataset : ', 'yeast'
        assert False

def test_yeastloadToMemoryTrue():
    utGenerator.run_test('yeast', True)

def test_yeastloadToMemoryFalse():
    utGenerator.run_test('yeast', False)

def tearDown():
    dataset_store.delete('yeast')
