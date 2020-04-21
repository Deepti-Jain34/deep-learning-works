# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:08:20 2020

@author: Vinsmon TP
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import chatbot_config as conf
import time
import numpy as np

class SequenceToSequences:
    
    def __init__(self,
                 rnn_size, 
                 num_layers,
                 quesion_vocabulary_size,
                 answer_vocabulary_size,
                 encoder_embedding_size,
                 decoder_embedding_size,
                 word_to_int_mapper,
                 batch_size,
                 epochs,
                 keep_prob,
                 min_learning_rate,
                 lr,
                 learning_rate_decay,
                 checkpoint):
        
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.quesion_vocabulary_size = quesion_vocabulary_size
        self.answer_vocabulary_size = answer_vocabulary_size
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size
        self.word_to_int_mapper = word_to_int_mapper
        self.batch_size = batch_size
        self.epochs = epochs
        self.keep_prob = keep_prob
        self.min_learning_rate = min_learning_rate
        self.lr = lr
        self.learning_rate_decay = learning_rate_decay
        self.checkpoint = checkpoint
        
        tf.reset_default_graph()
        
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='Input')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='Targets')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='Learning_Rate')
        self.keep_probability = tf.placeholder(dtype=tf.float32, name='Keep_Probability')
        self.sequence_length = tf.placeholder_with_default(conf.SEQUENCE_LENGTH, None, name = 'sequence_length')
        
        self.input_shape = tf.shape(self.inputs)
        #Maps a sequence of symbols to a sequence of embeddings.
        encoder_embedded_input = tf.contrib.layers.embed_sequence(ids=tf.reverse(self.inputs, [-1]),
                                                                  vocab_size=self.quesion_vocabulary_size + 1,
                                                                  embed_dim=self.encoder_embedding_size,
                                                                  initializer = tf.random_uniform_initializer(0, 1))
        
        self.encoder_state = self.get_encoder_state(encoder_embedded_input=encoder_embedded_input)
        
        self.decoder_embeddings_matrix = tf.Variable(tf.random_uniform([self.answer_vocabulary_size + 1, self.decoder_embedding_size], 0, 1))
        preprocessed_targets = self.process_decoder_input()
        self.decoder_embedded_input = tf.nn.embedding_lookup(self.decoder_embeddings_matrix, preprocessed_targets)
        
        with tf.variable_scope("decoding") as decoding_scope:
            
            self.decoder_cell = self.get_lstm_dropout_applied_layers()
            self.weights = tf.truncated_normal_initializer(stddev = 0.1)
            self.biases = tf.zeros_initializer()
            
            # Create a fully connected layer.
            self.output_function = lambda x : tf.contrib.layers.fully_connected(x,
                                                                                self.quesion_vocabulary_size,
                                                                                None,
                                                                                scope=decoding_scope,
                                                                                weights_initializer=self.weights,
                                                                                biases_initializer=self.biases)
            self.training_predictions = self.decode_training_set(decoding_scope)
            decoding_scope.reuse_variables()
            self.test_predictions = self.decode_test_set(decoding_scope)

        with tf.name_scope("optimization"):
            self.loss_error = tf.contrib.seq2seq.sequence_loss(self.training_predictions,
                                                          self.targets,
                                                          tf.ones([self.input_shape[0], self.sequence_length]))
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.gradients = self.optimizer.compute_gradients(self.loss_error)
            self.clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in self.gradients if grad_tensor is not None]
            self.optimizer_gradient_clipping = self.optimizer.apply_gradients(self.clipped_gradients)
        
        print("GRAPH IS DEFINED")
        
    def get_lstm_dropout_applied_layers(self):
        '''
        RNN cell composed sequentially of multiple lstm dropout applied simple cells.
        '''
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units= self.rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(cell=lstm, input_keep_prob=self.keep_probability)
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_dropout] * self.num_layers)
        return rnn_cell
    
    def get_encoder_state(self, encoder_embedded_input):
        encoder_cell = self.get_lstm_dropout_applied_layers()
        encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = self.sequence_length,
                                                                    inputs = encoder_embedded_input,
                                                                    dtype = tf.float32)
        return encoder_state
    
    def process_decoder_input(self):
        '''
        Preprocess the target label data for the training phase. 
        add <SOS> special token in front of all target data. 
        '''
        left_side = tf.fill([self.batch_size, 1], self.word_to_int_mapper['<SOS>'])
        right_side = tf.strided_slice(self.targets, [0,0], [self.batch_size, -1], [1,1])
        preprocessed_targets = tf.concat([left_side, right_side], 1)
        return preprocessed_targets
    
    def get_attention(self):
        attention_states = tf.zeros([self.batch_size, 1, self.decoder_cell.output_size])
        return tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                      attention_option = "bahdanau", 
                                                      num_units = self.decoder_cell.output_size)
        
    def decode_test_set(self, decoding_scope):
        attention_keys, attention_values, attention_score_function, attention_construct_function = self.get_attention()
        test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn=self.output_function,
                                                                                  encoder_state=self.encoder_state[0],
                                                                                  attention_keys=attention_keys,
                                                                                  attention_values=attention_values,
                                                                                  attention_score_fn=attention_score_function,
                                                                                  attention_construct_fn=attention_construct_function,
                                                                                  embeddings=self.decoder_embeddings_matrix,
                                                                                  start_of_sequence_id=self.word_to_int_mapper['<SOS>'],
                                                                                  end_of_sequence_id=self.word_to_int_mapper['<EOS>'],
                                                                                  maximum_length=self.sequence_length -1,
                                                                                  num_decoder_symbols=self.answer_vocabulary_size,
                                                                                  name = "attn_dec_inf")
        test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(self.decoder_cell,
                                                                                                                    test_decoder_function,
                                                                                                                    scope = decoding_scope)
        return test_predictions
    
    def decode_training_set(self, decoding_scope):
        attention_keys, attention_values, attention_score_function, attention_construct_function = self.get_attention()
        training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(self.encoder_state[0],
                                                                                  attention_keys,
                                                                                  attention_values,
                                                                                  attention_score_function,
                                                                                  attention_construct_function,
                                                                                  name = "attn_dec_train")
        decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(self.decoder_cell,
                                                                                                                  training_decoder_function,
                                                                                                                  self.decoder_embedded_input,
                                                                                                                  self.sequence_length,
                                                                                                                  scope = decoding_scope)
        decoder_output_dropout = tf.nn.dropout(decoder_output, self.keep_probability)
        return self.output_function(decoder_output_dropout)
    
    def split_into_batches_train(self, questions, answers, batch_size):
        for batch_index in range(0, len(questions) // batch_size):
            start_index = batch_index * batch_size
            questions_in_batch = questions[start_index : start_index + batch_size]
            answers_in_batch = answers[start_index : start_index + batch_size]
            yield questions_in_batch, answers_in_batch
    
    def split_into_batches_validation(self, questions, answers, batch_size):
        for batch_index in range(0, len(questions) // batch_size):
            start_index = batch_index * batch_size
            questions_in_batch = questions[start_index : start_index + batch_size]
            answers_in_batch = answers[start_index : start_index + batch_size]
            yield questions_in_batch, answers_in_batch
    
    def train(self, X_train, y_train, X_validation, y_validation, session=None):
        batch_index_check_training_loss = 100
        batch_index_check_validation_loss = ((len(X_train)) // self.batch_size // 2) - 1
        total_training_loss_error = 0
        list_validation_loss_error = []
        early_stopping_check = 0
        early_stopping_stop = 100
        
        if not session:
            session = tf.Session()
            session.run(tf.global_variables_initializer())
            print("using new session")
        else:
            print("session restored")
        
        for epoch in range(1, self.epochs + 1):
            for batch_index, (questions_in_batch, answers_in_batch) in enumerate(self.split_into_batches_train(X_train, y_train, self.batch_size)):    
                starting_time = time.time()
                _, batch_training_loss_error = session.run([self.optimizer_gradient_clipping, self.loss_error], {self.inputs: questions_in_batch,
                                                                                               self.targets: answers_in_batch,
                                                                                               self.learning_rate: self.lr,
                                                                                               self.sequence_length: answers_in_batch.shape[1],
                                                                                               self.keep_probability: self.keep_prob})
                total_training_loss_error += batch_training_loss_error
                ending_time = time.time()
                batch_time = ending_time - starting_time
                
                # check 100 batched done?
                if batch_index % batch_index_check_training_loss == 0 :
                    print('''Epoch: {:>3}/{}, 
                          Batch: {:>4}/{}, 
                          Training Loss Error: {:>6.3f}, 
                          Training Time on 100 Batches: {:d} seconds'''.format(epoch,self.epochs,batch_index,
                                                                               len(X_train) // self.batch_size,
                                                                               total_training_loss_error / batch_index_check_training_loss,
                                                                               int(batch_time * batch_index_check_training_loss)))
                    total_training_loss_error = 0
                if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
                    total_validation_loss_error = 0
                    starting_time = time.time()
                    for batch_index_validation, (questions_in_batch, answers_in_batch) in enumerate(self.split_into_batches_validation(X_validation, y_validation, self.batch_size)):
                        batch_validation_loss_error = session.run(self.loss_error, {self.inputs: questions_in_batch,
                                                                                    self.targets: answers_in_batch,
                                                                                    self.learning_rate: self.lr,
                                                                                    self.sequence_length: answers_in_batch.shape[1],
                                                                                    self.keep_probability: 1})
                        total_validation_loss_error += batch_validation_loss_error
                    ending_time = time.time()
                    batch_time = ending_time - starting_time
                    average_validation_loss_error = total_validation_loss_error / (len(X_validation) / self.batch_size)
                    print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
                    
                    self.lr *= self.learning_rate_decay
                    if self.lr < self.min_learning_rate: 
                        self.lr = self.min_learning_rate
                    list_validation_loss_error.append(average_validation_loss_error)
                    if average_validation_loss_error <= min(list_validation_loss_error):
                        print('I speak better now!!')
                        early_stopping_check = 0
                        saver = tf.train.Saver()
                        saver.save(session, self.checkpoint)
                    else:
                        print("Sorry I do not speak better, I need to practice more.")
                        early_stopping_check += 1
                        if early_stopping_check == early_stopping_stop:
                            break
            if early_stopping_check == early_stopping_stop:
                print("My apologies, I cannot speak better anymore. This is the best I can do.")
                break
            
    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print('Session restored')
        return sess
    
    def talk_to_me(self, question_encoded,session):
        fake_batch = np.zeros((self.batch_size, conf.SEQUENCE_LENGTH))
        fake_batch[0] = question_encoded
        predicted_answer = session.run(self.test_predictions, {self.inputs: fake_batch, self.keep_probability: 0.5})[0]
        answersints2word = {w_i: w for w, w_i in self.word_to_int_mapper.items()}
        answer = ''
        for i in np.argmax(predicted_answer, 1):
            if answersints2word[i] == 'i':
                token = ' I'
            elif answersints2word[i] == '<EOS>':
                token = '.'
            elif answersints2word[i] == '<OUT>':
                token = 'out'
            else:
                token = ' ' + answersints2word[i]
            answer += token
            if token == '.':
                break
        return answer