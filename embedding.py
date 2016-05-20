import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.engine.topology import Merge
from keras.layers.core import Flatten, Dense, Activation
from keras.optimizers import SGD

import math
import random

records = []
for line in open("ml-1m/ratings.dat"):
    userId, itemId, rating, timestamp = line.strip().split("::")
    records.append([int(userId), int(itemId), float(rating)])
random.shuffle(records)
records = np.array(records)

ratio = 0.9
recordsNum = records.shape[0]
splitPoint = int(recordsNum * ratio)

tr = records[:splitPoint,]
ts = records[splitPoint:,]

L = tr.shape[0]
M = ts.shape[0]

user_count = 6040 + 1
item_count = 3952 + 1

model_left = Sequential()
model_left.add(Embedding(user_count, 10, input_length=1))
model_right = Sequential()
model_right.add(Embedding(item_count, 10, input_length=1))
model = Sequential()
model.add(Merge([model_left, model_right], mode='concat'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
# model.add(Dense(64))
# model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.02, momentum=0., decay=0., nesterov=False))

model.fit(
    [tr[:,0].reshape((L,1)), tr[:,1].reshape((L,1))],
    tr[:,2].reshape((L,1)),
    batch_size=32,
    nb_epoch=20,
    validation_data=(
        [ts[:,0].reshape((M,1)), ts[:,1].reshape((M,1))],
        ts[:,2].reshape((M,1))
    )
)

result = model.predict([ts[:,0].reshape((M,1)), ts[:,1].reshape((M,1))])

err = 0
for i in range(M):
    err += (result[i][0] - ts[i][2]) ** 2
err = math.sqrt(err/M)
print("%.6f"%err)


