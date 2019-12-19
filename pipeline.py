#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:17:57 2019

@author: asem
"""

from pdd.datasets.grape import load_data

train_data_path, test_data_path = load_data(path='moss.tar', origin ="https://github.com/A-Alaa/pdd_new/raw/master/examples/moss-db.tar" , split_on_train_test=True, random_state=13)


from pdd.models import get_feature_extractor
from pdd.models import make_siamese
import tensorflow as tf


input_shape = (256, 256, 3)

print("Building feature extractor...")
feature_extractor = get_feature_extractor(input_shape)

print("Constructing siamese network...")
siams = make_siamese(feature_extractor, dist='l1', loss='cross_entropy')
siams.summary()


from pdd.utils.training import SiameseBatchGenerator

train_batch_gen = SiameseBatchGenerator.from_directory(dirname=train_data_path, augment=True)
test_batch_gen = SiameseBatchGenerator.from_directory(dirname=test_data_path)

def siams_generator(batch_gen, batch_size=None):
    while True:
        batch_xs, batch_ys = batch_gen.next_batch(batch_size, img_shape = input_shape)
        yield [batch_xs[0], batch_xs[1]], batch_ys
        


import time
start_time = time.time()

siams.fit_generator(
    generator=siams_generator(train_batch_gen),
    steps_per_epoch=100,
    epochs=100,
    verbose=1,
    validation_data=siams_generator(test_batch_gen),
    validation_steps=30,
    shuffle=True
)
# your code
elapsed_time = time.time() - start_time

#
#print("Saving feature extractor...")
#feature_extractor.save('pdd_feature_extractor.h5')
#
#
#from sklearn.metrics import accuracy_score
#from keras.models import load_model
#
#from pdd.models import TfKNN
#from pdd.utils.data_utils import create_dataset_from_dir
#
#
#
#print("Loading feature extractor...")
#feature_extractor = load_model("pdd_feature_extractor.h5")
#print("Loading datasets...")
#train_dataset = create_dataset_from_dir(train_data_path, shuffle=True)
#test_dataset = create_dataset_from_dir(test_data_path, shuffle=True)
#
#
#tfknn = TfKNN(feature_extractor, 
#              (train_dataset['data'], train_dataset['target']))
#
#
## predictions and similarities
#preds, sims = tfknn.predict(test_dataset['data'])
#accuracy = accuracy_score(test_dataset['target'], preds)
#print("Accuracy: %.2f" % accuracy)
#
#
#
#tfknn.save_graph_for_serving("tfknn_graph")
#
