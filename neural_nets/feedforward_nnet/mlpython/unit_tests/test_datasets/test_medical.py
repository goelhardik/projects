'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('medical')
    except:
        print 'Could not download the dataset : ', 'medical'
        assert False

def test_medicalloadToMemoryTrue():
    utGenerator.run_test('medical', True)

def test_medicalloadToMemoryFalse():
    utGenerator.run_test('medical', False)

def tearDown():
    dataset_store.delete('medical')
