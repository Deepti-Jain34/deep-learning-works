# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 08:06:42 2020

@author: Vinsmon TP
"""

import chatbot_config as config
import re
import nltk
import itertools
import numpy as np
import pickle

def read_data_to_lowercase(file:str):
    return open(file=file, encoding='utf-8').read().lower().split('\n')

def clean_text(text):
    text = str(text).strip()
    text = text.lower()
    text = re.sub(r"''+", '',text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"idk", "i dont know", text)
    text = re.sub(r"i'll", "i will", text)
    if(text.startswith("'")):
        text = text[1:]
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub("' +"," ",text)
    text = re.sub(r" 'the", ' the',  text)
    text = re.sub(r"'em ", ' them',  text)
    text = re.sub(r" 'em ", ' them',  text)
    text = re.sub(r" s ", ' ',  text)
    text = re.sub(r" i ", ' ',  text)
    text = re.sub(r" '", ' ',  text)
    text = re.sub(r" its ", ' it is ',  text)
    text = re.sub(r" abt ", ' about ',  text)
    text = re.sub(r" acct ", ' account ',  text)
    text = re.sub(r" ad ", ' adverticement ',  text)
    text = re.sub(r" ads ", ' adverticement ',  text)
    
    text = re.sub(r" ahh ", ' ah ',  text)
    text = re.sub(r" ahhh ", ' af ',  text)
    text = re.sub(r" ahhhh ", ' ah ',  text)
    text = re.sub(r" ahhhhh ", ' ah ',  text)
    text = re.sub(r" aint ", " ain't ",  text)
    text = re.sub(r" aka ", " also known as ",  text)
    text = re.sub(r"  +", ' ',text)
    text = str(text).strip()
    return text

def remove_blacklisted(line):
    return ''.join([ ch for ch in line if ch in config.CHAR_WHITELIST ])

def cleaned_data(data):
    cleaned_data=[]
    for line in data:
        cleaned_line = remove_blacklisted(line)
        cleaned_line = clean_text(cleaned_line)
        
        cleaned_data.append(cleaned_line)
    return cleaned_data

def get_qes_ans(data):
    questions, answers = [], []
    _data_len = len(data)
    sorted_questions, sorted_answers = [], []
    
    if _data_len % 2 != 0:
        _data_len = _data_len-1
        
    for _i in range(0, _data_len, 2):
        _qlen, _alen = len(data[_i].split(' ')), len(data[_i+1].split(' '))
        
        if _qlen == 1:
            if data[_i].isalpha():
                if _alen == 1:
                    if data[_i+1].isalpha() or data[_i+1].isdigit():
                        questions.append(data[_i])
                        answers.append( data[_i+1] + ' <EOS>' )
                    else:
                        pass
                else:
                    if _qlen >= config.MIN_QES_LEN and _qlen <= config.MAX_QES_LEN:
                        if _alen >= config.MIN_ANS_LEN and _alen <= config.MAX_ANS_LEN:
                            questions.append(data[_i])
                            answers.append( data[_i+1] + ' <EOS>' )
            else: 
                pass
        else:
            if _alen == 1:
                if data[_i+1].isalpha() or data[_i+1].isdigit():
                    if _qlen >= config.MIN_QES_LEN and _qlen <= config.MAX_QES_LEN:
                        if _alen >= config.MIN_ANS_LEN and _alen <= config.MAX_ANS_LEN:
                            questions.append(data[_i])
                            answers.append( data[_i+1] + ' <EOS>' )
                else:
                    pass
            else:
                if _qlen >= config.MIN_QES_LEN and _qlen <= config.MAX_QES_LEN:
                    if _alen >= config.MIN_ANS_LEN and _alen <= config.MAX_ANS_LEN:
                        questions.append(data[_i])
                        answers.append( data[_i+1] + ' <EOS>' )
                        
    for length in range(1, config.MAX_QES_LEN + 1):
        for i in enumerate(questions):
            if len(i[1]) == length:
                sorted_questions.append(questions[i[0]])
                sorted_answers.append(answers[i[0]])
                
    return sorted_questions, sorted_answers

def get_w2i_and_i2w(questions, answers):
    qtokenized = [ wordlist.split(' ') for wordlist in questions ]
    atokenized = [ wordlist.split(' ') for wordlist in answers ]
    tokenized_sentences = qtokenized + atokenized
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(config.VOCABULARY_SIZE)
    vocab = [ x[0] for x in vocab ] + ['<PAD>','<OUT>', '<SOS>']
    word2index = {word:i for i, word in enumerate(vocab)}
    index2word = {i:word for i, word in enumerate(vocab)}
    return index2word, word2index

def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = config.SEQUENCE_LENGTH
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

def encoder(data, word_to_int_mapper):
    questions_into_int = []
    for question in data:
        ints = []
        for word in question.split():
            if word not in word_to_int_mapper:
                ints.append(word_to_int_mapper['<OUT>'])
            else:
                ints.append(word_to_int_mapper[word])
        questions_into_int.append(ints)
    return np.array(apply_padding(questions_into_int, word_to_int_mapper))    

def preprocess_train_data():
    data_cleaned = cleaned_data(read_data_to_lowercase(config.DATA_PATH))
    quesions, answers = get_qes_ans(data_cleaned)
    index2word, word2index = get_w2i_and_i2w(quesions, answers)
    input_data = encoder(quesions, word2index)
    target_data = encoder(answers, word2index)
    np.save(config.QUES_PROCESSED_PATH,input_data)
    np.save(config.ANS_PROCESSED_PATH,target_data)
    metadata = {'w2idx' : word2index, 'idx2w' : index2word }
    with open(config.METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def load_data():
    with open(config.METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    input_q = np.load(config.QUES_PROCESSED_PATH)
    target_ans = np.load(config.ANS_PROCESSED_PATH)
    return input_q, target_ans, metadata['w2idx'], metadata['idx2w']

def train_test_split(input_Q, target_A, test_size=0.15):
    assert(len(input_Q) == len(target_A))
    _train_size = int(len(input_Q) * (1-test_size))
    return input_Q[0:_train_size], target_A[0:_train_size], input_Q[_train_size:], target_A[_train_size:]

