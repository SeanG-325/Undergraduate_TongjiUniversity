import tensorflow as tf
import numpy as np
import pandas as pd
from math import log as ln
from math import exp as exp

movies_df = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, engine='python')
movies_df.head()

ratings_df = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, engine='python')
ratings_df.head()

# Let's now rename the columns in these dataframes so we can better convey their data more intuitively:
movies_df.columns = ['MovieID', 'Title', 'Genres']
movies_df.head()

# And our final ratings_df:

ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df.head()
# The Restricted Boltzmann Machine model has two layers of neurons, one of which is what we call a visible input layer and the other is called a hidden layer. 
# The hidden layer is used to learn features from the information fed through the input layer. For our model, the input is going to contain X neurons, where X is the amount of movies in our dataset. 
# Each of these neurons will possess a normalized rating value varying from 0 to 1, where 0 meaning that a user has not watched that movie and the closer the value is to 1, the more the user likes the movie that neuron's representing. 
# These normalized values, of course, will be extracted and normalized from the ratings dataset.
# 
# After passing in the input, we train the RBM on it and have the hidden layer learn its features. 
# These features are what we use to reconstruct the input, which in our case, will predict the ratings for movies that user hasn't watched, which is exactly what we can use to recommend movies!
# 
# We will now begin to format our dataset to follow the model's expected input.

# First let's see how many movies we have and see if the movie ID's correspond with that value:
#len(movies_df)


# Now, we can start formatting the data into input for the RBM. We're going to store the normalized users ratings into as a matrix of user-rating called trX, and normalize the values.

user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
user_rating_df.head()

# extracting rating time matrix and fit it into rating matrix, and initialize time value
k = 1

user_rating_df_time = ratings_df.pivot(index='UserID', columns='MovieID', values='Timestamp')
user_rating_df_time.head()
time_df = user_rating_df_time.fillna(-1)
time = time_df.values
time_max = np.max(time, axis=1)
timesize = time.shape
for i in range(0, timesize[0]):
    for j in range(0, timesize[1]):
        if(time[i][j] != -1):
            time[i][j] = (time_max[i] - time[i][j])/(30*24*3600)
            #time[i][j] = 1 / (1 + k*ln(time[i][j] + 1)) - 0.5
            time[i][j] = 0.5 + 0.5 * exp(-time[i][j])

# Lets normalize it now:

norm_user_rating_df = user_rating_df.fillna(0) / 5.0 #归一化
trX = norm_user_rating_df.values

# Next, let's start building our RBM with TensorFlow. 
#We'll begin by first determining the number of neurons in the hidden layers and then creating placeholder variables for storing our visible layer biases, hidden layer biases and weights that connects the hidden layer with the visible layer. 
#We will be arbitrarily setting the number of neurons in the hidden layers to 20. You can freely set this value to any number you want since each neuron in the hidden layer will end up learning a feature.

hiddenUnits = 200
visibleUnits =  len(user_rating_df.columns)

with tf.name_scope('bias'):
    vb = tf.placeholder("float", [visibleUnits]) #Number of unique movies
    hb = tf.placeholder("float", [hiddenUnits]) #Number of features we're going to learn
    W = tf.placeholder("float", [visibleUnits, hiddenUnits])


# We then move on to creating the visible and hidden layer units and setting their activation functions. 
# In this case, we will be using the tf.sigmoid and tf.relu functions as nonlinear activations since it is commonly used in RBM's.

#Phase 1: Input Processing
    t = tf.placeholder("float", [visibleUnits])

with tf.name_scope('Input'):
    v0 = tf.placeholder("float", [None, visibleUnits])
    #v0 = tf.multiply(v0, t)
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
#Phase 2: Reconstruction
with tf.name_scope('Reconstruct'):
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb + t) 
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    #v1 = tf.multiply(v1, t)
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)


# Now we set the RBM training parameters and functions.

#Learning rate
alpha = 0.4
#Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)
update_t = t + alpha * tf.reduce_mean(v0 - v1, 0)


# And set the error function, which in this case will be the Mean Absolute Error Function.
with tf.name_scope('Error'):
    err = v0 - v1
    err_sum = tf.reduce_mean(err * err)
    tf.summary.scalar('err_sum', err_sum)

#Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)
#Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

cur_t = np.zeros([visibleUnits], np.float32)

#Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)
#Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)

prv_t = np.zeros([visibleUnits], np.float32)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

# Now we train the RBM with 20 epochs with each epoch using 10 batches with size 100. After training, we print out a graph with the error by epoch.

epochs = 100
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        timebatch = time[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb, t:prv_t})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb, t:prv_t})
        cur_nb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb, t:prv_t})
        cur_t = sess.run(update_t, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb, t:prv_t})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        prv_t = cur_t
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb, t:prv_t}))
    print ('epoch {}: {}'.format(i,errors[-1]))

# We can now predict movies that an arbitrarily selected user might like. 
# This can be accomplished by feeding in the user's watched movie preferences into the RBM and then reconstructing the input. 
# The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched based on the preferences of the users that the RBM was trained on.

# Lets first select a <b>User ID</b> of our mock user:

mock_user_id = 215

#Selecting the input user
inputUser = trX[mock_user_id-1].reshape(1, -1)
inputUser[0:5]

#Feeding in the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb + t)
feed = sess.run(hh0, feed_dict={ v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb, t:prv_t})
print(rec)


# We can then list the 20 most recommended movies for our mock user by sorting it by their scores given by our model.

scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
print(scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20))


# So, how to recommend the movies that the user has not watched yet? 

# Now, we can find all the movies that our mock user has watched before:

movies_df_mock = ratings_df[ratings_df['UserID'] == mock_user_id]
print(movies_df_mock.head())


# In the next cell, we merge all the movies that our mock users has watched with the predicted scores based on his historical data:

#Merging movies_df with ratings_df by MovieID
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock, on='MovieID', how='outer')


# lets sort it and take a look at the first 20 rows:

print(merged_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20))