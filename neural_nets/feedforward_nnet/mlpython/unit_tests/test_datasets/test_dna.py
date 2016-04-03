'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('dna')
    except:
        print 'Could not download the dataset : ', 'dna'
        assert False

def test_dnaloadToMemoryTrue():
    utGenerator.run_test('dna', True)

def test_dnaloadToMemoryFalse():
    utGenerator.run_test('dna', False)

def tearDown():
    dataset_store.delete('dna')
