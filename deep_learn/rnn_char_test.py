# -*- coding: utf-8 -*-
"""
Created on Sat May 21 14:34:08 2016

@author: yangsicong
"""

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

text = open('./input.txt', 'r').read()
char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
vocab_size = len(char_to_idx)

print('Working on %d characters (%d unique)' % (len(text), vocab_size))

SEQ_LENGTH = 64
BATCH_SIZE = 16
BATCH_CHARS = len(text) / BATCH_SIZE
LSTM_SIZE = 512
LAYERS = 3

def read_batches(text):
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    X = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
    Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))

    for i in range(0, BATCH_CHARS - SEQ_LENGTH - 1, SEQ_LENGTH):
        X[:] = 0
        Y[:] = 0
        for batch_idx in range(BATCH_SIZE):
            start = batch_idx * BATCH_CHARS + i
            for j in range(SEQ_LENGTH):
                X[batch_idx, j, T[start+j]] = 1
                Y[batch_idx, j, T[start+j+1]] = 1

        yield X, Y


def build_model(batch_size, seq_len):
    model = Sequential()
    model.add(LSTM(LSTM_SIZE, return_sequences=True, batch_input_shape=(batch_size, seq_len, vocab_size), stateful=True))
    model.add(Dropout(0.2))
    for l in range(LAYERS - 1):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model


print ('Building model.')
test_model = build_model(1, 1)
training_model = build_model(BATCH_SIZE, SEQ_LENGTH)
print ('... done')

def sample(epoch, sample_chars=256):
    test_model.reset_states()
    test_model.load_weights('./tmp/keras_char_rnn.%d.h5' % epoch)
    header = 'LSTM based '
    sampled = [char_to_idx[c] for c in header]

    for c in header:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, char_to_idx[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=softmax)
        sampled.append(sample)

    print (''.join([idx_to_char[c] for c in sampled]))

for epoch in range(100):
    for i, (x, y) in enumerate(read_batches(text)):
        loss = training_model.train_on_batch(x, y)
        print (epoch, i, loss)

        if i % 1000 == 0:
            training_model.save_weights('./tmp/keras_char_rnn.%d.h5' % epoch, overwrite=True)
            sample(epoch)