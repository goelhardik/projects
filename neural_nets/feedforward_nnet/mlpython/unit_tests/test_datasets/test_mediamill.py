'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mediamill')
    except:
        print 'Could not download the dataset : ', 'mediamill'
        assert False

def test_mediamillloadToMemoryTrue():
    utGenerator.run_test('mediamill', True)

def test_mediamillloadToMemoryFalse():
    utGenerator.run_test('mediamill', False)

def tearDown():
    dataset_store.delete('mediamill')
