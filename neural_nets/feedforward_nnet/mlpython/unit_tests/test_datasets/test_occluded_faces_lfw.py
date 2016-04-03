'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('occluded_faces_lfw')
    except:
        print 'Could not download the dataset : ', 'occluded_faces_lfw'
        assert False

def test_occluded_faces_lfwloadToMemoryTrue():
    utGenerator.run_test('occluded_faces_lfw', True)

def test_occluded_faces_lfwloadToMemoryFalse():
    utGenerator.run_test('occluded_faces_lfw', False)

def tearDown():
    dataset_store.delete('occluded_faces_lfw')
