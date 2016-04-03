'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('bibtex')
    except:
        print 'Could not download the dataset : ', 'bibtex'
        assert False

def test_bibtexloadToMemoryTrue():
    utGenerator.run_test('bibtex', True)

def test_bibtexloadToMemoryFalse():
    utGenerator.run_test('bibtex', False)

def tearDown():
    dataset_store.delete('bibtex')
