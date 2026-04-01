# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:40:16 2021

@author: Ranak Roy Chowdhury
"""

# -*- coding: utf-8 -*-

import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dimensions = 2
data = []
label = None

for i in range(1, dimensions + 1):
    filename = f'AtrialFibrillationDimension{i}_TEST.arff'

    dataset = arff.load(filename)                  # returns generator in your setup
    dataset = np.array(list(dataset), dtype=object)

    print("dataset shape:", dataset.shape)
    print("first row type:", type(dataset[0]))

    data.append(dataset[:, :-1].astype(np.float32))

    if label is None:
        label = np.array(dataset[:, -1])

data = np.array(data, dtype=np.float32)            # (dimensions, samples, seq_len)
data = np.transpose(data, (1, 2, 0))               # (samples, seq_len, dimensions)

print("X_test shape:", data.shape)
np.save('X_test.npy', data)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label).reshape(-1, 1)

try:
    onehot_encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print("y_test shape:", onehot_encoded.shape)
np.save('y_test.npy', onehot_encoded)