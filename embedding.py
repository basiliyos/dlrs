import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.engine.topology import Merge
from keras.layers.core import Flatten, Dense, Activation

import math

tr = []
for line in open("train.txt"):
    t = line.strip().split(" ")
    tr.append([int(t[0]), int(t[1]), float(t[2])])
tr = np.array(tr)

ts = []
for line in open("test.txt"):
    t = line.strip().split(" ")
    ts.append([int(t[0]), int(t[1]), float(t[2])])
ts = np.array(ts)

movie_count = 3706
user_count = 6040
model_left = Sequential()
model_left.add(Embedding(movie_count, 60, input_length=1))
model_right = Sequential()
model_right.add(Embedding(user_count, 20, input_length=1))
model = Sequential()
model.add(Merge([model_left, model_right], mode='concat'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adadelta')

L = 900189
M = 100020

model.fit(
    [tr[:,1].reshape((L,1)), tr[:,0].reshape((L,1))],
    tr[:,2].reshape((L,1)),
    batch_size=24000,
    nb_epoch=42,
    validation_data=(
        [ts[:,1].reshape((M,1)), ts[:,0].reshape((M,1))],
        ts[:,2].reshape((M,1))
    )
)

result = model.predict([ts[:,1].reshape((M,1)), ts[:,0].reshape((M,1))])

err = 0
for i in range(len(ts)):
    err += (result[i][0] - ts[i][2]) ** 2
err = math.sqrt(err/len(ts))
print("%.6f"%err)
