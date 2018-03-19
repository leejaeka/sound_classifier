from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_utils import load_features, load_features_with_deltas_stacking,random_mini_batches


# In[2]:


#train_data, test_data, train_labels, test_labels = load_features()
# now with deltas
train_data, test_data, train_labels, test_labels = load_features_with_deltas_stacking()
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)


# In[3]:


train_data.shape


# In[4]:


test_data.shape


# In[5]:


train_labels.shape


# In[6]:


test_labels.shape


# In[7]:


size_input=train_data.shape[1]*train_data.shape[2]


# In[8]:


size_input


# In[9]:


#Seeding
seed = 3

# Network Parameters
num_input = size_input
num_classes = 2
dropout = 0.50

# tf Graph input
X = tf.placeholder(tf.float32, [None, train_data.shape[1], train_data.shape[2]])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
beta1 = tf.Variable(0.9, name = 'beta1', dtype=tf.float32, trainable=False)
beta2 = tf.Variable(0.999, name = 'beta2', dtype=tf.float32, trainable=False)
epsilon = tf.Variable(1e-08, name = 'epsilon', dtype=tf.float32, trainable=False)


# In[10]:


train_data.shape[1], train_data.shape[2]


# In[11]:


# Model
def sound_net(x, weights, biases, dropout):
    # Input Layer
    x = tf.reshape(x, shape=[-1, train_data.shape[1], train_data.shape[2], 1])

    # Convolutional Layer #1
    #input shape [batch, train_data.shape[1], train_data.shape[2], 1]
    #output shape [batch, train_data.shape[1], train_data.shape[2], 80]   
    conv1 = tf.layers.conv2d(inputs=x, filters=80, kernel_size=[57, 6], padding="same", activation=tf.nn.relu)

    # Pooling Layer #1
    # Input Tensor Shape: [batch_size, train_data.shape[1], train_data.shape[2], 80]
    # Output Tensor Shape: [batch_size, train_data.shape[1]/2, train_data.shape[2]/2, 80]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="same", strides=2)
    pool1 = tf.nn.dropout(pool1,dropout)
    
    # Convolutional Layer #2
    #input shape [batch_size, train_data.shape[1]/2, train_data.shape[2]/2, 80]
    #output shape [batch_size, train_data.shape[1]/2, train_data.shape[2]/2, 80]
    conv2 = tf.layers.conv2d(inputs=pool1, filters=80, kernel_size=[1, 3], padding="same", activation=tf.nn.relu)
    
    # Pooling Layer #2
    # Input Tensor Shape: [batch_size, train_data.shape[1]/2, train_data.shape[2]/2, 80]
    # Output Tensor Shape: [batch_size, train_data.shape[1]/2/8, train_data.shape[2]/2/7, 2, 80]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[8, 7], padding="same", strides=[8, 7])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    dense1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.add(tf.matmul(dense1, weights['wd1']), biases['bd1'])
    dense1 = tf.nn.relu(dense1)
    # Apply Dropout
    dense1 = tf.nn.dropout(dense1,dropout)
   
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    dense2 = tf.reshape(dense1, [-1, weights['wd2'].get_shape().as_list()[0]])
    dense2 = tf.add(tf.matmul(dense2, weights['wd2']), biases['bd2'])
    dense2 = tf.nn.relu(dense2)
    # Apply Dropout
    dense2=tf.nn.dropout(dense2,dropout)
   
    # Output, class prediction
    out = tf.add(tf.matmul(dense2, weights['out']), biases['out'])
    return out


# In[12]:


int(train_data.shape[1]/2/8*train_data.shape[2]/2/7*80)


# In[13]:


# Store weights and biases
weights = {
    # 57x6 conv
    'wc1': tf.Variable(tf.random_normal([57, 6, 1, 80])),
    # 1x3 conv
    'wc2': tf.Variable(tf.random_normal([1, 3, 80, 80])),
    # fully connected, 8*2*80 inputs (after maxpool), 1024 outputs
    'wd1': tf.Variable(tf.random_normal([int(train_data.shape[1]/2/8*train_data.shape[2]/2/7*80), 1024])),
    # fully connected, 1014 inputs (after maxpool), 1024 outputs
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    # 1024 inputs, 2 outputs
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([80])),
    'bc2': tf.Variable(tf.random_normal([80])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# In[14]:


# build model
logits = sound_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize
init = tf.global_variables_initializer()


# In[15]:


def run_model(num_epochs, train_data, train_labels, test_data, test_labels, minibatch_size, seed, learn_rate, actual_beta1, actual_beta2, actual_epsilon):
    costs = []
    with tf.Session() as sess:
                    
        sess.run(init)
        #Override defaults for Adam
        sess.run(beta1.assign(actual_beta1))
        sess.run(beta2.assign(actual_beta2))
        sess.run(epsilon.assign(actual_epsilon))        
        actual_beta1 = sess.run(beta1)  
        actual_beta2 = sess.run(beta2)  
        actual_epsilon = sess.run(epsilon)    

        for epoch in list(range(int(num_epochs))):

            epoch_cost = 0.
            seed = seed + 1
            minibatches = random_mini_batches(train_data, train_labels, minibatch_size, seed)

            for minibatch in minibatches:
                (batch_x, batch_y) = minibatch
                #convert to one_hot for labels
                batch_y = tf.one_hot(batch_y, num_classes)
                batch_y = sess.run(batch_y)
                batch_y = batch_y.reshape((batch_y.shape[0], num_classes))

                sess.run(train_op, feed_dict={X: batch_x, 
                                              Y: batch_y, 
                                              keep_prob: dropout, 
                                              learning_rate: learn_rate, 
                                              beta1: actual_beta1,
                                              beta2: actual_beta2,
                                              epsilon: actual_epsilon})
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0,
                                                                     learning_rate: learn_rate, 
                                                                     beta1: actual_beta1,
                                                                     beta2: actual_beta2,
                                                                     epsilon: actual_epsilon})
                epoch_cost += loss

            #After running all minibatches    
            if epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)

        #Plot loss over time
    #     plt.plot(np.squeeze(costs))
    #     plt.ylabel('cost')
    #     plt.xlabel('iterations (per tens)')
    #     plt.title("Learning rate =" + str(learning_rate))
    #     plt.show()  
    
        # Train accuracy
        # train_labels = tf.one_hot(train_labels, num_classes)
        # train_labels = sess.run(train_labels)    
        # print("Train Accuracy:", \
            # sess.run(accuracy, feed_dict={X: train_data,
                                          # Y: train_labels,
                                          # keep_prob: 1.0}))

        # Test accuracy
        test_labels = tf.one_hot(test_labels, num_classes)
        test_labels = sess.run(test_labels)    
        print("Test Accuracy:",             sess.run(accuracy, feed_dict={X: test_data,
                                          Y: test_labels,
                                          keep_prob: 1.0}))


# In[16]:

thislearning_rate=0.0001
print('learning rate',thislearning_rate)

run_model(num_epochs=100, train_data=train_data, train_labels=train_labels, 
          test_data=test_data, test_labels=test_labels, 
          minibatch_size=64, 
          learn_rate=thislearning_rate, seed=seed, actual_beta1=0.9, actual_beta2=0.999, actual_epsilon=1e-08)


# only run line below if you want to reset and restart

# In[17]:


#tf.reset_default_graph()

