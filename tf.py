import tensorflow as tf
import numpy as np
import random

records = []
for line in open("ml-1m/ratings.dat"):
    user_id, item_id, rating, timestamp = line.strip().split("::")
    records.append([int(user_id), int(item_id), float(rating)])
#random.shuffle(records)
#records = np.array(records)

#ratio = 0.9
#records_count = records.shape[0]
#split_point = int(records_count * ratio)

#train = records[:split_point,]
#test  = records[split_point:,]

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
W2 = tf.Variable(tf.random_uniform([20, 10], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([1]))
b2 = tf.Variable(tf.zeros([1]))
b3 = tf.Variable(tf.zeros([1]))

y1 = tf.sigmoid(tf.matmul(merge_model, W1)  + b1)
y2 = tf.sigmoid(tf.matmul(y1, W2) + b2)
y = tf.matmul(y2, W3) + b3

loss = tf.reduce_mean(tf.square(y - rating))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

batch_size = 16

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for step in xrange(200):
        print(step)
        random.shuffle(records)
        data = np.array(records)[:batch_size,]
        train.run(feed_dict={
            user_id : np.reshape( data[:,0] , (batch_size, 1) ),
            item_id : np.reshape( data[:,1] , (batch_size, 1) ),
            rating  : np.reshape( data[:,2] , (batch_size, 1) )
        })
        

