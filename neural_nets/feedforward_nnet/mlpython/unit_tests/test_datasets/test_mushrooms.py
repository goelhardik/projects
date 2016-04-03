'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mushrooms')
    except:
        print 'Could not download the dataset : ', 'mushrooms'
        assert False

def test_mushroomsloadToMemoryTrue():
    utGenerator.run_test('mushrooms', True)

def test_mushroomsloadToMemoryFalse():
    utGenerator.run_test('mushrooms', False)

def tearDown():
    dataset_store.delete('mushrooms')
