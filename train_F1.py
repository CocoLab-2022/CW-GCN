from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

from utils import *
from models_our_target import GCN, MLP
from sklearn.metrics import f1_score


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('source_name', 'citationv1', 'Dataset string.') # acmv9 citationv1 dblpv7
flags.DEFINE_string('target_name', 'acmv9', 'Dataset string.') # acmv9 citationv1 dblpv7
flags.DEFINE_string('data_folder', './data/', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('learning_rate2', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def get_mean(embedding, weight):
    sum_w = np.sum(weight)

    weight = np.multiply(weight, 1 / sum_w)
    mean = 0
    f = []
    for i in range(embedding.shape[0]):
        f.append(embedding[i] * weight[i])
    f_new = np.array(f, dtype=np.float32)
    mean = np.sum(f_new, 0)

    var = np.zeros((1, embedding.shape[1]), dtype=np.float32)
    for i in range(embedding.shape[0]):
        var = var + np.power((embedding[i] - mean), 2) * weight[i]

    V = np.sum(np.power(weight, 2))

    std = np.power(var / (1 - V), 1 / 2)

    return mean, std

# Define model evaluation function
def evaluate(sess,model,features, support,labels, mask, mean_s, std_s, mean_s_2, std_s_2, embedding_1, embedding_2,
                 placeholders):
        def small_trick(y_test, y_pred):
            y_pred_new = np.zeros(y_pred.shape, np.int32)
            sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
            for i in range(y_test.shape[0]):
                num = int(sum(y_test[i]))
                for j in range(num):
                    y_pred_new[i][sort_index[i][j]] = 1
            return y_pred_new

        t_test = time.time()
        feed_dict_val = construct_feed_dict_target(features, support,labels, mask, mean_s, std_s, mean_s_2, std_s_2,
                                            embedding_1, embedding_2, placeholders)
        outs_val = sess.run([model.outputs, model.accuracy], feed_dict=feed_dict_val)

        y_pred = 1 / (1 + np.exp(-outs_val[0]))
        y_pred = small_trick(labels, y_pred)
        micro = f1_score(labels, y_pred, average="micro")
        macro = f1_score(labels, y_pred, average="macro")

        return micro, macro, (time.time() - t_test)


def train_F1(FLAGS):
    tf.reset_default_graph()

    adj, features, y_train, y_val, train_mask, val_mask, adj_target, features_target, y_target, target_mask = load_clean_data(
        FLAGS.source_name, FLAGS.target_name, FLAGS.data_folder)

    pkl_file = open('./saved_model/feature1.pkl', 'rb')

    embedding_1 = pkl.load(pkl_file)

    embedding_1 = embedding_1[0]

    embedding_1 = np.array(embedding_1, dtype='float64');

    pkl_file2 = open('./saved_model/feature2.pkl', 'rb')

    embedding_2 = pkl.load(pkl_file2)
    embedding_2 = embedding_2[0]

    pkl_file3 = open('./saved_model/weight.pkl', 'rb')

    weight = pkl.load(pkl_file3)
    weight = weight[0]


    mean_s, std_s = get_mean(embedding_1, weight)

    mean_s_2, std_s_2 = get_mean(embedding_2, weight)

    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    features_target = preprocess_features(features_target)
    if FLAGS.model == 'gcn':
        support_target = [preprocess_adj(adj_target)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support_target = chebyshev_polynomials(adj_target, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support_target = [preprocess_adj(adj_target)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'mean_s': tf.placeholder(tf.float32),
        'std_s': tf.placeholder(tf.float32),
        'mean_s_2': tf.placeholder(tf.float32),
        'std_s_2': tf.placeholder(tf.float32),
        'embedding_1': tf.placeholder(tf.float32, shape=(None, embedding_1.shape[1])),
        'embedding_2': tf.placeholder(tf.float32, shape=(None, embedding_2.shape[1]))
    }


    # Create model
    model = model_func(placeholders, input_dim=features[2][1], source_num=embedding_1.shape[0],
                       target_num=adj_target.shape[0], logging=True)

    # Initialize session
    # sess = tf.Session()

    save_path = "./saved_model/"

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model.restore_model(sess, save_path)

    cost_val = []
    cost = 0
    best_micro = 0
    best_macro = 0


    for epoch in range(FLAGS.epochs):


        # Construct feed dictionary
        feed_dict = construct_feed_dict_target(features_target,support_target,y_target, target_mask, mean_s,
                                        std_s, mean_s_2, std_s_2, embedding_1, embedding_2, placeholders)


        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

        cost_val.append(outs[1])


        if (epoch + 1) % 100 == 0:
            micro, macro, duration = evaluate(sess,model,features_target, support_target, y_target, target_mask, mean_s,
                                              std_s, mean_s_2, std_s_2, embedding_1, embedding_2, placeholders)

            print("Epoch:", '%04d' % (epoch + 1),
                  "target_micro=", "{:.5f}".format(micro),
                  "target_macro=", "{:.5f}".format(macro))
            if micro > best_micro:
                best_micro = micro
                best_macro = macro



        if epoch >= FLAGS.early_stopping and cost_val[-1] > np.mean(
                cost_val[-(FLAGS.early_stopping + 1):-1]):  # np.abs(outs[-1]-cost) <= 0.000001:

            last_micro, last_macro, duration = evaluate(sess, model, features_target, support_target,y_target,
                                              target_mask, mean_s,std_s, mean_s_2, std_s_2, embedding_1, embedding_2, placeholders)

            print("target_micro=", "{:.5f}".format(best_micro),"target_macro=", "{:.5f}".format(best_macro))
            print("last_target_micro=", "{:.5f}".format(last_micro),"last_target_macro=", "{:.5f}".format(last_macro))
            print("Early stopping...")
            break
if __name__ == "__main__":
    train_F1(FLAGS)