# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:45:48 2020

@author: Vinsmon TP
"""

## DATA PREPERATION CONFIGURATIONS ##
MIN_QES_LEN = 1
MAX_QES_LEN = 30
MIN_ANS_LEN = 1
MAX_ANS_LEN = MAX_QES_LEN - 1 

VOCABULARY_SIZE = 6000
CHAR_WHITELIST="'0123456789abcdefghijklmnopqrstuvwxyz "
UNK='unk'
DATA_PATH='./data/chat.txt'
QUES_PROCESSED_PATH = './data/preprocessed/encoded_qestions.npy'
ANS_PROCESSED_PATH = './data/preprocessed/encoded_answers.npy'
METADATA_PATH = './data/preprocessed/metadata.pkl'

## MODEL PARAMETERS

SEQUENCE_LENGTH = MAX_QES_LEN
RNN_SIZE = 1024
NUM_LAYERS = 3
ENCODER_EMBEDDING_SIZE = 1024
DECODER_EMBEDDING_SIZE = 1024
BATCH_SIZE = 32
EPOCHS = 2
KEEP_PROB = .5
MIN_LEARNING_RATE = 0.001
LR = 0.001
LEARNING_RATE_DECAY = 1
CHECK_POINT_PATH =  "./data/checkpoints/"