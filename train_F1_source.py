from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import pickle as pkl

from sklearn.metrics import f1_score
from utils import *
from models_our import GCN, MLP
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('source_name', 'citationv1', 'Dataset string.') # acmv9 citationv1 dblpv7
flags.DEFINE_string('target_name', 'citationv1', 'Dataset string.') # acmv9 citationv1 dblpv7
flags.DEFINE_string('data_folder', './data/', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('learning_rate2', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
all_acc_val = []


def evaluate(sess,model,features, support,labels, mask, placeholders):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape, np.int32)
        sort_index = np.flip(np.argsort(y_pred, axis=1), 1)

        for i in range(y_test.shape[0]):
            num = int(sum(y_test[i]))
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = 1
        return y_pred_new

    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support,labels, mask, placeholders)
    outs_val = sess.run([model.outputs, model.accuracy], feed_dict=feed_dict_val)

    y_pred = 1 / (1 + np.exp(-outs_val[0]))
    y_pred = small_trick(labels, y_pred)
    micro = f1_score(labels, y_pred, average="micro")
    macro = f1_score(labels, y_pred, average="macro")

    return micro, macro, (time.time() - t_test)
def train_F1_source(FLAGS):
    tf.reset_default_graph()
    adj, features, y_train, y_val, train_mask, val_mask, adj_target, features_target, y_target, target_mask = load_noisy_data(
        FLAGS.source_name, FLAGS.target_name, FLAGS.data_folder)

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


    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight': tf.placeholder(tf.float32),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    }
    #

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Define model evaluation function

    save_path = "./saved_model/"

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    micro_old = 0.

    for epoch in range(FLAGS.epochs):
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.accuracy], feed_dict=feed_dict)
        if (epoch + 1) % 10 == 0:
            micro, macro, duration = evaluate(sess,model,features, support, y_train, train_mask, placeholders)
            # Print results
            # cost_val.append(cost)
            if micro >= micro_old:
                micro_old = micro
                print("Epoch:", '%04d' % (epoch + 1),
                  "train_micro=", "{:.5f}".format(micro),
                  "train_macro=", "{:.5f}".format(macro), "time=", "{:.5f}".format(duration))

            if micro >= 0.95:
                model.save_model(sess, save_path + '/epoch' + str(epoch) + ".model")
                Embedding_1 = sess.run([model.graph_P1_0], feed_dict=feed_dict)
                Embedding_2 = sess.run([model.graph_P1], feed_dict=feed_dict)
                pkl.dump(Embedding_1, open("./saved_model/feature1.pkl", "wb"))

                pkl.dump(Embedding_2, open("./saved_model/feature2.pkl", "wb"))
                weight = sess.run([model.weight], feed_dict=feed_dict)
                pkl.dump(weight, open("./saved_model/weight.pkl", "wb"))

            if micro >= 0.95 and macro >= 0.95:
                print("Early stopping...")
                break

if __name__ =="__main__":
    train_F1_source(FLAGS)