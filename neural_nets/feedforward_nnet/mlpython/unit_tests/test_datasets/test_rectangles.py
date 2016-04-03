'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('rectangles')
    except:
        print 'Could not download the dataset : ', 'rectangles'
        assert False

def test_rectanglesloadToMemoryTrue():
    utGenerator.run_test('rectangles', True)

def test_rectanglesloadToMemoryFalse():
    utGenerator.run_test('rectangles', False)

def tearDown():
    dataset_store.delete('rectangles')
