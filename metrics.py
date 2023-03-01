import tensorflow as tf
import numpy as np




def L1_loss(W):

    d=tf.square(W)
    loss =1- tf.exp(tf.div(-d, 1))
    return tf.reduce_sum(loss)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    
#    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels),1)
    p=tf.nn.sigmoid(preds)
    print(labels.shape)
    print(preds.shape)
    distance=tf.reduce_sum(tf.square(p - labels),1)


    loss =1- tf.exp(tf.div(-distance, 100)) 

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask

    return tf.reduce_mean(loss)



def masked_correntropy(preds, labels, mask, weight):
    """Softmax cross-entropy loss with masking."""

    distance = tf.reduce_sum(tf.square(preds-labels),1)
    loss =1- tf.exp(tf.div(-distance, 1))
#    mask = 1-mask
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    #loss *=weight
    return tf.reduce_mean(loss)


def masked_link_loss(preds, adj, mask, weight):
    """Softmax cross-entropy loss with masking."""
#    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#    loss = tf.reduce_sum(1- tf.exp(tf.div(-tf.pow((preds - labels),2), 10)))
    #adj = tf.sparse_tensor_to_dense(adj)
    
    def get_cos_distance(X1, X2):
    # calculate cos distance between two sets
    # more similar more big
      # (k,n) = X1.shape
      # (m,n) = X2.shape
    
       X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
       X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
    
       X1_X2 = tf.matmul(X1, tf.transpose(X2))
       X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[3327,1]),tf.reshape(X2_norm,[1,3327]))
    
       cos = X1_X2/X1_X2_norm
       return cos


    A=tf.sigmoid(tf.matmul(preds,tf.transpose(preds)))
    A=tf.multiply(A,adj)


    loss = tf.reduce_sum(tf.square(A-adj),1)
#    loss =1- tf.exp(tf.div(-distance, 5))
    mask = 1-mask
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
#    loss *= mask
    #loss *=weight
    return tf.reduce_mean(loss)
   

def link_loss(X, XS, S, adj, mask, weight):

    def cosine_similarity(vector1, vector2):

    # normalize input
      norm_vector1 = tf.nn.l2_normalize(vector1, 0)
      norm_vector2 = tf.nn.l2_normalize(vector2, 0)

    # multiply row i with row j using transpose
      similarity = tf.reduce_sum(tf.multiply(norm_vector1, norm_vector2))

      return similarity

    loss =tf.reduce_sum(tf.square(X - XS))
        
    S=tf.add(S,tf.transpose(S))/2
    adj = tf.sparse_tensor_to_dense(adj)
    #dim=adj.get_shape()[1]
    #x=tf.sparse_tensor_dense_matmul(adj,tf.eye(dim))
    adj=tf.reshape(adj, [2708,2708])
    
    for i in range(2708):
      # print(S[i,:].get_shape())
      # print(adj[i,:].get_shape())

       loss +=cosine_similarity(S[:,i],adj[:,i])
    return loss

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
