'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('majmin')
    except:
        print 'Could not download the dataset : ', 'majmin'
        assert False

def test_majminloadToMemoryTrue():
    utGenerator.run_test('majmin', True)

def test_majminloadToMemoryFalse():
    utGenerator.run_test('majmin', False)

def tearDown():
    dataset_store.delete('majmin')
