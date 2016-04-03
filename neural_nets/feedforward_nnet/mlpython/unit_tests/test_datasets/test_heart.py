'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('heart')
    except:
        print 'Could not download the dataset : ', 'heart'
        assert False

def test_heartloadToMemoryTrue():
    utGenerator.run_test('heart', True)

def test_heartloadToMemoryFalse():
    utGenerator.run_test('heart', False)

def tearDown():
    dataset_store.delete('heart')
