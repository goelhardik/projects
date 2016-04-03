'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('rcv1')
    except:
        print 'Could not download the dataset : ', 'rcv1'
        assert False

def test_rcv1loadToMemoryTrue():
    utGenerator.run_test('rcv1', True)

def test_rcv1loadToMemoryFalse():
    utGenerator.run_test('rcv1', False)

def tearDown():
    dataset_store.delete('rcv1')
