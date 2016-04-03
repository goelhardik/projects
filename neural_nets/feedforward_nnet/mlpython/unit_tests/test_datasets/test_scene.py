'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('scene')
    except:
        print 'Could not download the dataset : ', 'scene'
        assert False

def test_sceneloadToMemoryTrue():
    utGenerator.run_test('scene', True)

def test_sceneloadToMemoryFalse():
    utGenerator.run_test('scene', False)

def tearDown():
    dataset_store.delete('scene')
