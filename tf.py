import tensorflow as tf
import numpy as np
import random

records = []
for line in open("ml-1m/ratings.dat"):
    user_id, item_id, rating, timestamp = line.strip().split("::")
    records.append([int(user_id), int(item_id), float(rating)])
records = np.array(records)
np.random.shuffle(records)

ratio = 0.9
records_count = records.shape[0]
split_point = int(records_count * ratio)

train = records[:split_point,]
test  = records[split_point:,]

user_count = 6040 + 1
item_count = 3952 + 1

embedding_size = 20
user_embeddings = tf.Variable(tf.random_uniform([user_count, embedding_size], -1.0, 1.0))
item_embeddings = tf.Variable(tf.random_uniform([item_count, embedding_size], -1.0, 1.0))

user_id = tf.placeholder(tf.int32, [None, 1])
item_id = tf.placeholder(tf.int32, [None, 1])
rating  = tf.placeholder(tf.float32, [None, 1])

user_model = tf.nn.embedding_lookup(user_embeddings, user_id)
item_model = tf.nn.embedding_lookup(item_embeddings, item_id)
merge_model = tf.concat(2, [user_model, item_model])
merge_model = tf.reshape(merge_model, [-1, embedding_size * 2])

W1 = tf.Variable(tf.random_uniform([embedding_size * 2, 20], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([20, 1], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([1]))
b2 = tf.Variable(tf.zeros([1]))

y1 = tf.sigmoid(tf.matmul(merge_model, W1)  + b1)
y = tf.matmul(y1, W2) + b2

mse = tf.reduce_mean(tf.square(y - rating))
rmse = tf.sqrt(mse)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(mse)

epoch_count = 20
batch_size = 32

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in xrange(1, epoch_count+1):
        np.random.shuffle(train)

        for i in range(train.shape[0]/batch_size):
            data = train[i*batch_size:i*batch_size+batch_size,]
        
            train_step.run(feed_dict={
                user_id : data[:,0:1],
                item_id : data[:,1:2],
                rating  : data[:,2:3]
            })
    
        result = rmse.eval(feed_dict={
            user_id : test[:,0:1],
            item_id : test[:,1:2],
            rating  : test[:,2:3]
        })
    
        print(epoch, result)
