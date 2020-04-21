# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:29:49 2020

@author: Vinsmon TP
"""

import data_preprocessor as dp
import VanilaSquenceToSequence as seq2seq
import chatbot_config as conf

encoded_input_data, encoded_output_data, w2idx, idx2w = dp.load_data()
X_train, y_train, X_test, y_test = dp.train_test_split(encoded_input_data, encoded_output_data, test_size=.2)

rnn_size = conf.RNN_SIZE
num_layers = conf.NUM_LAYERS
quesion_vocabulary_size = len(w2idx)
answer_vocabulary_size = len(idx2w)
encoder_embedding_size = conf.ENCODER_EMBEDDING_SIZE
decoder_embedding_size = conf.DECODER_EMBEDDING_SIZE
word_to_int_mapper = w2idx
batch_size = conf.BATCH_SIZE
epochs = conf.EPOCHS
keep_prob = conf.KEEP_PROB
min_learning_rate = conf.MIN_LEARNING_RATE
lr = conf.LR
learning_rate_decay = conf.LEARNING_RATE_DECAY
checkpoint = conf.CHECK_POINT_PATH

model = seq2seq.SequenceToSequences(rnn_size,
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
                                    checkpoint)

model.train(X_train, y_train, X_test, y_test, model.restore_last_session())

session = model.restore_last_session()

question = dp.encoder(["i fuck you"], word_to_int_mapper)
answe = model.talk_to_me(question,session)
print(answe)

