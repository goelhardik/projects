'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('convex')
    except:
        print 'Could not download the dataset : ', 'convex'
        assert False

def test_convexloadToMemoryTrue():
    utGenerator.run_test('convex', True)

def test_convexloadToMemoryFalse():
    utGenerator.run_test('convex', False)

def tearDown():
    dataset_store.delete('convex')
