'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('cadata')
    except:
        print 'Could not download the dataset : ', 'cadata'
        assert False

def test_cadataloadToMemoryTrue():
    utGenerator.run_test('cadata', True)

def test_cadataloadToMemoryFalse():
    utGenerator.run_test('cadata', False)

def tearDown():
    dataset_store.delete('cadata')
