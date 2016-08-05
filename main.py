import tensorflow as tf
import numpy as np
import json
import reader
import os

class Model():
    def __init__(self):
        self.sentence_embed_size = 500
        self.batchsize = 32
        self.vocabsize = (10 * 1000) + 2
        self.word_embed_size = 300
        self.sentence_length = 30
        self.dropout_prob = 1.0
        self.numlayers = 1
        self.z_size = 500
        self.encoder_hidden_size = 500
        self.decoder_hidden_size = 500
        self.max_gradient_norm = 5.0

        # self.word_embed_size = 3
        # self.encoder_hidden_size = 5
        # self.decoder_hidden_size = 5
        # self.sentence_embed_size = 5
        # self.z_size = 5
        # self.sentence_length = 4
        # self.vocabsize = 3

        self.sentences_in = tf.placeholder(tf.int32, [self.batchsize, self.sentence_length])
        self.sentences_in_decoded = tf.placeholder(tf.int32, [self.batchsize, self.sentence_length])
        self.latentscale = tf.placeholder(tf.float32)

        self.z_mean, self.z_stddev = self.encoder()
        samples = tf.random_normal([self.batchsize,self.z_size],0,1,dtype=tf.float32)
        self.z = self.z_mean + (self.z_stddev * samples)
        # self.z = self.debug_out
        # self.z = tf.ones([self.batchsize, self.z_size])
        self.d = self.decoder()

        flat_in = tf.reshape(self.sentences_in,[self.batchsize * self.sentence_length,1])
        # flat_d = tf.reshape(self.d, [self.batchsize * self.sentence_length, self.vocabsize])
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(flat_d, flat_in)
        flat_d = tf.reshape(self.d, [self.batchsize * self.sentence_length, self.decoder_hidden_size])
        cross_entropy = tf.nn.sampled_softmax_loss(tf.transpose(self.d_w2), self.d_b2, flat_d, flat_in, 512, self.vocabsize)
        self.generation_loss = tf.reduce_sum(tf.reshape(cross_entropy, [self.batchsize, self.sentence_length]), reduction_indices=1)
        # self.generation_loss = tf.nn.l2_loss(self.sentences_guessed_flattened - self.sentences_in_flattened) / (self.sentence_embed_size * self.sentence_length)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(tf.square(self.z_stddev)) - 1,1)
        #
        self.cost = tf.reduce_mean(self.generation_loss + self.latentscale*self.latent_loss)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
        self.optim = tf.train.AdamOptimizer(0.0001)
        self.update = self.optim.apply_gradients(zip(clipped_gradients,params))

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def encoder(self):
        with tf.variable_scope("encoder"):
            one_cell = tf.nn.rnn_cell.BasicLSTMCell(self.sentence_embed_size)
            one_cell = tf.nn.rnn_cell.DropoutWrapper(one_cell,self.dropout_prob)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell] * self.numlayers)
            self.encoder_initial_state = lstm_cell.zero_state(self.batchsize, tf.float32)

            self.encoder_embedding = tf.get_variable("embedding", [self.vocabsize, self.word_embed_size])
            inputs = tf.nn.embedding_lookup(self.encoder_embedding, self.sentences_in)
            list_format = [tf.squeeze(x, [1]) for x in tf.split(1, self.sentence_length, inputs)]
            outputs, last_state = tf.nn.seq2seq.rnn_decoder(list_format, self.encoder_initial_state, lstm_cell, loop_function=None, scope='encoder')

            # for now let's assume all sentences have 30 words
            output = outputs[self.sentence_length-1]
            self.debug_out = output

            w1 = tf.get_variable("w1",[self.sentence_embed_size,self.encoder_hidden_size])
            b1 = tf.get_variable("b1",[self.encoder_hidden_size])
            w2 = tf.get_variable("w2",[self.encoder_hidden_size,self.encoder_hidden_size])
            b2 = tf.get_variable("b2",[self.encoder_hidden_size])

            w_mean = tf.get_variable("w_mean", [self.encoder_hidden_size, self.z_size])
            b_mean = tf.get_variable("b_mean", [self.z_size])

            w_stddev = tf.get_variable("w_stddev", [self.encoder_hidden_size, self.z_size])
            b_stddev = tf.get_variable("b_stddev", [self.z_size])

            h1 = tf.nn.sigmoid(tf.matmul(output,w1) + b1)
            h2 = tf.nn.sigmoid(tf.matmul(h1,w2) + b2)

            o_mean = tf.matmul(h2,w_mean) + b_mean
            o_stddev = tf.matmul(h2,w_stddev) + b_stddev

            return o_mean, o_stddev

    def decoder(self):
        with tf.variable_scope("decoder"):
            one_cell = tf.nn.rnn_cell.BasicLSTMCell(self.sentence_embed_size)
            one_cell = tf.nn.rnn_cell.DropoutWrapper(one_cell,self.dropout_prob)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell] * self.numlayers)
            self.decoder_initial_state = lstm_cell.zero_state(self.batchsize, tf.float32)

            def loop(prev, _):
                return prev


            self.encoder_embedding = tf.get_variable("embedding", [self.vocabsize, self.word_embed_size])
            inputs = tf.nn.embedding_lookup(self.encoder_embedding, self.sentences_in_decoded)
            list_format = [tf.concat(1, [tf.squeeze(x, [1]), self.z]) for x in tf.split(1, self.sentence_length, inputs)]

            outputs, last_state = tf.nn.seq2seq.rnn_decoder(list_format, self.encoder_initial_state, lstm_cell, loop_function=None, scope='decoder')

            w1 = tf.get_variable("w1",[self.sentence_embed_size,self.decoder_hidden_size])
            b1 = tf.get_variable("b1",[self.decoder_hidden_size])
            self.d_w2 = tf.get_variable("w2",[self.decoder_hidden_size,self.vocabsize])
            self.d_b2 = tf.get_variable("b2",[self.vocabsize])

            self.flattened_output = tf.reshape(outputs, [self.sentence_length * self.batchsize, self.sentence_embed_size])

            self.d_h1 = tf.nn.sigmoid(tf.matmul(self.flattened_output,w1) + b1)
            # self.h2 = tf.matmul(h1,w2) + b2

            reshaped_output = tf.reshape(self.d_h1, [self.sentence_length, self.batchsize, self.decoder_hidden_size])
            switched_output = tf.transpose(reshaped_output, [1, 0, 2])

            return switched_output

    def train(self):
        fakedata = np.zeros((2,4))
        fakedata[0,:] = [1,1,0,0]
        fakedata[1,:] = [2,2,0,0]


        # for i in xrange(1000):
        #     guess, z, z_mean, z_stddev, gen_loss, latent_loss, _ = self.sess.run([self.d, self.z, self.z_mean, self.z_stddev, self.generation_loss, self.latent_loss, self.optimizer], feed_dict={self.sentences_in: fakedata})
        #     print "%f %f" % (np.mean(gen_loss), np.mean(latent_loss))
        #     print np.argmax(guess,axis=2)
        #     # print z_mean
        #     # print z_stddev
        #     print z
        #     # print partway.shape
        #     np.set_printoptions(threshold=np.inf)

        raw_data = reader.ptb_raw_data("/home/kevin/Documents/Datasets/simple-examples/data")
        train_data, valid_data, test_data, vocabsize = raw_data
        print vocabsize
        # print train_data

        list(reader.ptb_iterator(valid_data, self.batchsize, self.sentence_length))



        saver = tf.train.Saver(max_to_keep=2)
        # saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        ls = 0.1
        for epoch in xrange(10000):
            if epoch > 20:
                ls = min(1, epoch / 50.0)

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
                gen_loss, latent_loss, _ = self.sess.run([self.generation_loss, self.latent_loss, self.update], feed_dict={self.sentences_in: x, self.sentences_in_decoded: x2, self.latentscale: ls})
                gl = np.mean(gen_loss) / self.sentence_length
                # print "gen loss: %f latent loss: %f perplexity: %f" % (gl, np.mean(latent_loss), np.exp(gl))
                total_genloss += gl
                total_latentloss += np.mean(latent_loss)
                steps = steps + 1
            print "epoch %d genloss %f perplexity %f latentloss %f" % (epoch, total_genloss / steps, np.exp(total_genloss/steps), total_latentloss)
            total_validloss = 0
            validsteps = 0
            for step, x in enumerate(reader.ptb_iterator(valid_data, self.batchsize, self.sentence_length)):
                x2 = np.copy(x)
                c = np.zeros((self.batchsize,1), dtype=np.int32)
                c.fill(10001)
                x = np.hstack((x[:,1:],c))
                # x: input
                # x2: desired output
                gen_loss, latent_loss = self.sess.run([self.generation_loss, self.latent_loss], feed_dict={self.sentences_in: x, self.sentences_in_decoded: x2, self.latentscale: ls})
                gl = np.mean(gen_loss) / self.sentence_length
                # print "gen loss: %f latent loss: %f perplexity: %f" % (gl, np.mean(latent_loss), np.exp(gl))
                total_validloss += gl
                validsteps = validsteps + 1
            print "valid %d genloss %f perplexity %f" % (epoch, total_validloss / validsteps, np.exp(total_validloss/validsteps))

            if epoch % 10 == 0:
                saver.save(self.sess, os.getcwd()+"/training-reg/train",global_step=epoch)



model = Model()
model.train()
