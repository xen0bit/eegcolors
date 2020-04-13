from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

model = tf.keras.models.load_model('./model/eegcolors.h5')

# Check its architecture
model.summary()

# column order in CSV file
column_names = ['TP9','AF7','AF8','TP10','Right AUX','Color']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

test_url = "https://raw.githubusercontent.com/xen0bit/eegcolors/master/merged_labelled_csv/test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

batch_size = 1

def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='Color',
    num_epochs=1,
    shuffle=False)

print(test_dataset)
test_dataset = test_dataset.map(pack_features_vector)
print(test_dataset)

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    #print(x)
    logits = model(x, training=False)
    #print(logits)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    #print(prediction)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))