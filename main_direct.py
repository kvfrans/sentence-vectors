import tensorflow as tf
import numpy as np
import json
import reader
import os

class Model():
    def __init__(self):
        self.batchsize = 32
        self.vocabsize = (10 * 1000) + 2
        self.word_embed_size = 300
        self.sentence_length = 30
        self.dropout_prob = 10
        self.num_layers = 1
        self.decoder_hidden_size = 500
        self.max_gradient_norm = 5.0
        self.sentence_embed_size = 500

        self.sentences_in = tf.placeholder(tf.int32, [self.batchsize, self.sentence_length])
        self.sentences_in_decoded = tf.placeholder(tf.int32, [self.batchsize, self.sentence_length])
        self.d = self.decoder()

        flat_in = tf.reshape(self.sentences_in, [self.batchsize * self.sentence_length,1])
        flat_d = tf.reshape(self.d, [self.batchsize * self.sentence_length, self.decoder_hidden_size])
        cross_entropy = tf.nn.sampled_softmax_loss(tf.transpose(self.d_w2), self.d_b2, flat_d, flat_in, 512, self.vocabsize)
        self.generation_loss = tf.reduce_sum(tf.reshape(cross_entropy, [self.batchsize, self.sentence_length]), reduction_indices=1)
        self.cost = tf.reduce_mean(self.generation_loss)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.optim = tf.train.AdamOptimizer(0.0001)
        self.update = self.optim.apply_gradients(zip(clipped_gradients, params))

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def decoder(self):
        with tf.variable_scope("decoder"):
            one_cell = tf.nn.rnn_cell.BasicLSTMCell(self.sentence_embed_size)
            one_cell = tf.nn.rnn_cell.DropoutWrapper(one_cell, self.dropout_prob)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell] * self.num_layers)
            self.decoder_initial_state = lstm_cell.zero_state(self.batchsize, tf.float32)

            self.decoder_embedding = tf.get_variable("embedding", [self.vocabsize, self.word_embed_size])
            inputs = tf.nn.embedding_lookup(self.decoder_embedding, self.sentences_in_decoded)
            list_format = [tf.squeeze(x,[1]) for x in tf.split(1, self.sentence_length, inputs)]

            outputs, last_state = tf.nn.seq2seq.rnn_decoder(list_format, self.decoder_initial_state, lstm_cell, loop_function=None, scope="decoder")

            output = tf.reshape(tf.concat(1,outputs), [self.sentence_length * self.batchsize, self.sentence_embed_size])

            w1 = tf.get_variable("w1", [self.sentence_embed_size, self.decoder_hidden_size])
            b1 = tf.get_variable("b1", [self.decoder_hidden_size])
            self.d_w2 = tf.get_variable("w2", [self.decoder_hidden_size, self.vocabsize])
            self.d_b2 = tf.get_variable("b2", [self.vocabsize])

            self.flattened_output = tf.reshape(outputs, [self.sentence_length * self.batchsize, self.sentence_embed_size])
            self.d_h1 = tf.nn.sigmoid(tf.matmul(self.flattened_output, w1) + b1)

            reshaped_output = tf.reshape(self.d_h1, [self.sentence_length, self.batchsize,self.decoder_hidden_size])
            switched_output = tf.transpose(reshaped_output, [1, 0, 2])

            return switched_output

    def train(self):
        raw_data = reader.ptb_raw_data("/home/kevin/Documents/Datasets/simple-examples/data")
        train_data, valid_data, test_data, vocabsize = raw_data
        print vocabsize

        saver = tf.train.Saver(max_to_keep=2)
        for epoch in xrange(10000):
            total_genloss = 0
            total_latentloss = 0
            steps = 0
            for step, x in enumerate(reader.ptb_iterator(test_data, self.batchsize, self.sentence_length)):
                x2 = np.copy(x)
                c = np.zeros((self.batchsize,1), dtype=np.int32)
                c.fill(10001)
                x = np.hstack((x[:,1:],c))
                # x: input
                # x2: desired output
                gen_loss, _ = self.sess.run([self.generation_loss, self.update], feed_dict={self.sentences_in: x, self.sentences_in_decoded: x2})
                gl = np.mean(gen_loss) / self.sentence_length
                total_genloss += gl
                steps = steps + 1
            print "epoch %d genloss %f perplexity %f" % (epoch, total_genloss / steps, np.exp(total_genloss/steps))
            total_validloss = 0
            validsteps = 0
            for step, x in enumerate(reader.ptb_iterator(valid_data, self.batchsize, self.sentence_length)):
                x2 = np.copy(x)
                c = np.zeros((self.batchsize,1), dtype=np.int32)
                c.fill(10001)
                x = np.hstack((x[:,1:],c))
                # x: input
                # x2: desired output
                gen_loss, _ = self.sess.run([self.generation_loss, self.update], feed_dict={self.sentences_in: x, self.sentences_in_decoded: x2})
                gl = np.mean(gen_loss) / self.sentence_length
                total_validloss += gl
                validsteps = validsteps + 1
            print "valid %d genloss %f perplexity %f" % (epoch, total_validloss / validsteps, np.exp(total_validloss/validsteps))

model = Model()
model.train()
