'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('adult')
    except:
        print 'Could not download the dataset : ', 'adult'
        assert False

def test_adultloadToMemoryTrue():
    utGenerator.run_test('adult', True)

def test_adultloadToMemoryFalse():
    utGenerator.run_test('adult', False)

def tearDown():
    dataset_store.delete('adult')
