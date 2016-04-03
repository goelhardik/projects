'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('connect4')
    except:
        print 'Could not download the dataset : ', 'connect4'
        assert False

def test_connect4loadToMemoryTrue():
    utGenerator.run_test('connect4', True)

def test_connect4loadToMemoryFalse():
    utGenerator.run_test('connect4', False)

def tearDown():
    dataset_store.delete('connect4')
