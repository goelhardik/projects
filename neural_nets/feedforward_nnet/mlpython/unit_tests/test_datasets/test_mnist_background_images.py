'''This file was generate with generatorPythonUnitTest.py'''
import mlpython.datasets.store as dataset_store
import os
from nose.tools import *
import utGenerator

def setUp():
    try:
        dataset_store.download('mnist_background_images')
    except:
        print 'Could not download the dataset : ', 'mnist_background_images'
        assert False

def test_mnist_background_imagesloadToMemoryTrue():
    utGenerator.run_test('mnist_background_images', True)

def test_mnist_background_imagesloadToMemoryFalse():
    utGenerator.run_test('mnist_background_images', False)

def tearDown():
    dataset_store.delete('mnist_background_images')
