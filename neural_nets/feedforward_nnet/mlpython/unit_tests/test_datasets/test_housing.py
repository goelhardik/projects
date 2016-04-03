'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('housing')
    except:
        print 'Could not download the dataset : ', 'housing'
        assert False

def test_housingloadToMemoryTrue():
    utGenerator.run_test('housing', True)

def test_housingloadToMemoryFalse():
    utGenerator.run_test('housing', False)

def tearDown():
    dataset_store.delete('housing')
