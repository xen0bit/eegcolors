from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
from time import time, sleep, strftime, gmtime
from muselsl.stream import find_muse
from muselsl import muse
from muselsl.constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK
import tensorflow as tf
from pprint import pprint
import math

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

model = tf.keras.models.load_model('./model/eegcolors.h5')

# Check its architecture
model.summary()

# column order in CSV file
column_names = ['TP9','AF7','AF8','TP10','Right AUX','Color']
colors = ['black', 'blue', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

feature_names = column_names[:-1]
label_name = column_names[-1]

colorRank = {
    'black' : 0,
    'blue' : 0,
    'green' : 0,
    'orange' : 0,
    'pink' : 0,
    'purple' : 0,
    'red' : 0,
    'white' : 0,
    'yellow' : 0
}

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

def pack_features_vector(features):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features

def predictColor(filename=None, dejitter=False, data_source="EEG"):
    chunk_length = LSL_EEG_CHUNK
    if data_source == "PPG":
        chunk_length = LSL_PPG_CHUNK
    if data_source == "ACC":
        chunk_length = LSL_ACC_CHUNK
    if data_source == "GYRO":
        chunk_length = LSL_GYRO_CHUNK

    print("Looking for a %s stream..." % (data_source))
    streams = resolve_byprop('type', data_source, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        print("Can't find %s stream." % (data_source))
        return

    print("Started acquiring data.")
    inlet = StreamInlet(streams[0], max_chunklen=chunk_length)
    # eeg_time_correction = inlet.time_correction()

    print("Looking for a Markers stream...")
    marker_streams = resolve_byprop(
        'name', 'Markers', timeout=LSL_SCAN_TIMEOUT)

    if marker_streams:
        inlet_marker = StreamInlet(marker_streams[0])
    else:
        inlet_marker = False
        print("Can't find Markers stream.")

    info = inlet.info()
    description = info.desc()

    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    res = []
    timestamps = []
    markers = []
    t_init = time()
    time_correction = inlet.time_correction()
    print('Start recording at time t=%.3f' % t_init)
    print('Time correction: ', time_correction)
    while True:
        res = []
        timestamps = []
        markers = []
        t_init = time()
        time_correction = inlet.time_correction()
        try:
            data, timestamp = inlet.pull_chunk(timeout=1.0,
                                               max_samples=chunk_length)

            if timestamp:
                res.append(data)
                timestamps.extend(timestamp)
            if inlet_marker:
                marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                if timestamp:
                    markers.append([marker, timestamp])
        except KeyboardInterrupt:
            break

        time_correction = inlet.time_correction()
        #print('Time correction: ', time_correction)

        res = np.concatenate(res, axis=0)
        timestamps = np.array(timestamps) + time_correction

        if dejitter:
            y = timestamps
            X = np.atleast_2d(np.arange(0, len(y))).T
            lr = LinearRegression()
            lr.fit(X, y)
            timestamps = lr.predict(X)

        res = np.c_[res]
        data = pd.DataFrame(data=res, columns=ch_names)

        if inlet_marker:
            n_markers = len(markers[0][0])
            for ii in range(n_markers):
                data['Marker%d' % ii] = 0
            # process markers:
            for marker in markers:
                # find index of markers
                ix = np.argmin(np.abs(marker[1] - timestamps))
                for ii in range(n_markers):
                    data.loc[ix, 'Marker%d' % ii] = marker[0][ii]

        # directory = os.path.dirname(filename)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        #data.to_csv(filename, float_format='%.3f', index=False)
        #print(data)
        test_dataset = tf.data.Dataset.from_tensor_slices(data.values)
        #test_dataset = list(test_dataset.as_numpy_iterator()) 

        #test_dataset = test_dataset.map(pack_features_vector)
        #print(list(test_dataset))
        for x in test_dataset:
            if (x.ndim == 1):
                x = np.array([x])
            logits = model(x, training=False)
            #print(logits)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            colorPrediction = colors[int(prediction[0])]
            colorRank[colorPrediction]+=1
            #pprint(colorRank)
            
        data = None
        if(math.floor(time()) % 2 == 0):
            #Pretty print ranks
            pprint(colorRank)
            #Clear ranks
            for i in colors:
                colorRank[i] = 0
        sleep(1)

        #print('Done - wrote file: ' + filename + '.')

predictColor()