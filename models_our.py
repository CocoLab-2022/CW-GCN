from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.loss2 = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.opt_op_2 = None

        self.layers_f = []
        self.P1 = []
        self.P2 = []
        self.P3 = []
        self.weight = None
        self.graph_P1 = None
        self.labels = None
        self.P1_out = None
        self.P2_out = None
        self.W0 = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self.saved_list = list(self.layers[0].vars.values()) + list(self.layers_f[0].vars.values()) + list(
            self.layers[1].vars.values())

        self.graph_P1_0 = self.layers[0](self.inputs)
        self.graph_P1 = self.layers[1](self.graph_P1_0)

        self.outputs = self.layers_f[0](self.graph_P1)  # (tf.concat([self.P1_out,self.P2_out],1))

        self.P2_out = self.P2[0](self.outputs)

        self.P3_out = self.P3[0](self.labels)

        p = tf.nn.sigmoid(self.outputs)

        distance = tf.reduce_sum(tf.square(p - self.placeholders['labels']), 1)

        self.weight = tf.exp(tf.div(-distance, 10))

        # Build metrics
        self._loss()
        self._accuracy()

        # class_list= list( self.layers[1].vars.values())#+list( self.layers[1].vars.values())

        self.opt_op = self.optimizer.minimize(self.loss)  # , var_list=class_list)

        self.opt_op_2 = self.optimizer.minimize(self.loss2)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def save_model(self, sess, path):
        saver = tf.train.Saver(var_list=self.saved_list)
        saver.save(sess, path)

    def restore_model(self, sess, path):
        def assign(a, b):
            op = a.assign(b)
            self.session.run(op)

        weights = tf.train.latest_checkpoint(path)
        saver = tf.train.Saver(var_list=self.saved_list)
        # saver = tf.train.Saver({"W0":self.W_source[0],  "W1":self.W_source[1], "W2":self.W_source[2], "W3":self.W_source[3], "W4":self.W_source[4],"W5":self.W_source[5],"b0":self.b_source[0],"b1":self.b_source[1],"b2":self.b_source[2],"b3":self.b_source[3],"b4":self.b_source[4],"b5":self.b_source[5]})#out_embed_layer":self.out_embed_layer})
        saver.restore(sess, weights)
        # assign(self.W_target[1], self.W_source[1])

        self.is_Init = True

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=False,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        print(tf.shape(self.inputs))
        self.labels = placeholders['labels']
        self.w = placeholders['weight']

        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        # self.opt_op = None

        self.build()

    def _loss(self):
        # Weight decay loss

        print(self.layers[0].vars.values())
        for var in self.layers[0].vars.values():
            print(var)
            #            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += 1 * masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=int(FLAGS.hidden1 / 2),
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        self.layers_f.append(Dense(input_dim=int(FLAGS.hidden1 / 2),
                                   output_dim=self.output_dim,
                                   placeholders=self.placeholders,
                                   act=lambda x: x,
                                   dropout=False,
                                   logging=self.logging))

        # self.graph_P1_2=self.layers[-1](self.graph_P1)
        self.P1.append(Dense(input_dim=FLAGS.hidden1,
                             output_dim=5,
                             placeholders=self.placeholders,
                             act=lambda x: x,
                             bias=False,
                             dropout=False,
                             logging=self.logging))

        self.P2.append(Dense(input_dim=self.output_dim,
                             output_dim=5,
                             placeholders=self.placeholders,
                             act=lambda x: x,
                             bias=False,
                             dropout=False,
                             logging=self.logging))

        self.P3.append(Dense(input_dim=self.output_dim,
                             output_dim=15,
                             placeholders=self.placeholders,
                             act=lambda x: x,
                             bias=False,
                             dropout=False,
                             logging=self.logging))


    def predict(self):
        return tf.nn.softmax(self.outputs)

