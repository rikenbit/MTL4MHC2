#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os
import random
import warnings
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, BatchNormalization, Dropout, GaussianNoise, GaussianDropout
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import CSVLogger, History
import keras.backend as backend
from tensorflow.python.keras.utils.vis_utils import plot_model
from datetime import datetime
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import layers
from keras_multi_head import MultiHead
import datetime
import pickle


# input file

c1_train = "{class I MHC train data set path}"
c1_val ="{class I MHC validation data set path}"

c2_train = "{class II MHC train data set path}"
c2_val = "{class II MHC validation data set path}"

# pkl file are available from https://github.com/rikenbit/MTL4MHC2/tree/main/dict

with open("{Path_to_pkl_file}/monovec.pkl","rb") as f:
     monovec = pickle.load(f)
    
with open("{Path_to_pkl_file}/trigram_to_idx_MHC.pkl","rb") as f:
    trigram_to_idx_MHC = pickle.load(f)

with open("{Path_to_pkl_file}/monogram_to_idx.pkl","rb") as f:
    monogram_to_idx = pickle.load(f)

with open("{Path_to_pkl_file}/trivec1_MHC.pkl","rb") as f:
    trivec1_MHC = pickle.load(f)  

# function

def replace(raw_seq_0):
    B_aa = 'DN'
    J_aa = 'IL'
    Z_aa = 'EQ'
    X_aa = 'ACDEFGHIKLMNPQRSTVWY'
    
    seq = raw_seq_0.str.replace('B', random.choice(B_aa))
    seq = seq.str.replace('J', random.choice(J_aa))
    seq = seq.str.replace('Z', random.choice(Z_aa))
    seq = seq.str.replace('X', random.choice(X_aa))
    raw_seq_0 = seq
    
    return raw_seq_0

# monogram
def monogram(raw_seq_0):
    feature_0 = []
    for i in range(0, len(raw_seq_0)):
        strain_embedding = []
        for j in range(0, len(raw_seq_0[i])):
            monogram = raw_seq_0[i][j]
            mono_embedding = monogram_to_idx["".join(monogram)]
            strain_embedding.append(mono_embedding)
            
        feature_0.append(strain_embedding)
    return feature_0

# trigram
def trigram(raw_seq_0):
    feature_0 = []
    for i in range(0, len(raw_seq_0)):
        strain_embedding = []
        for j in range(0, len(raw_seq_0[i]) - 2):
            trigram = raw_seq_0[i][j:j + 3]
            tri_embedding = trigram_to_idx_MHC["".join(trigram)]
            strain_embedding.append(tri_embedding)
            
        feature_0.append(strain_embedding)
    return feature_0


# model

def multi_head(out_dim, dropoutrate, lay_num, loss1, loss2):
    shared_embedding = Embedding(47, 100, weights=[monovec], trainable=False)
    shared_multi_head = MultiHead(
        layer=keras.layers.Bidirectional(keras.layers.LSTM(out_dim, dropout=dropoutrate), name='LSTM'),
        layer_num=lay_num,
        reg_index=[1, 4],
        reg_slice=(slice(None, None), slice(256, 192)),
        reg_factor=0.1,
        name='Multi-Head-Attention1',
    )
    
    text_input_c1 = keras.Input(shape=(None,))
    text_input_c2 = keras.Input(shape=(None,))
    
    encoded_input_c1 = shared_embedding(text_input_c1)
    encoded_input_c2 = shared_embedding(text_input_c2)
    
    # encoder1
    encoder_output1 = shared_multi_head(encoded_input_c1)
    peptide_output1 = keras.layers.Flatten(name='Flatten1')(encoder_output1)
    out1 = Dense(2, activation='softmax')(peptide_output1)
    
    # encoder2
    encoder_output2 = shared_multi_head(encoded_input_c2)
    MHC_output1 = keras.layers.Flatten(name='Flatten2')(encoder_output2)
    out2 = Dense(2, activation='softmax')(MHC_output1)

    model = Model([text_input_c1, text_input_c2], outputs=[out1, out2])  # ??????????????????????????????Model??????????????????????????????
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy',], loss_weights=[loss1, loss2], optimizer="adam", metrics=['accuracy', tp, tn, fp, fn])
    
    return model



    
def bilstmc2_model(out_dim, dropoutrate, lay_num):
    encoder_input = Input(shape=(39,)) # encoder????????????
    decoder_input = Input(shape=(230,)) # decoder????????????
    
    encoder_emb = Embedding(47, 100, input_length=39, weights=[monovec], trainable=False)(encoder_input)
    decoder_emb = Embedding(9419, 100, input_length=230, weights=[trivec1_MHC], trainable=False)(decoder_input)
    
    # encoder
    encoder_output1 = MultiHead(
        layer=keras.layers.Bidirectional(keras.layers.LSTM(out_dim, dropout=dropoutrate), name='LSTM'),
        layer_num=lay_num,
        reg_index=[1, 4],
        reg_slice=(slice(None, None), slice(128, 96)),
        reg_factor=0.1,
        name='Multi-Head-Attention1',
    )(encoder_emb)
    peptide_output1 = keras.layers.Flatten(name='Flatten1')(encoder_output1)
    
    # decoder
    decoder_output1 = MultiHead(
        layer=keras.layers.Bidirectional(keras.layers.LSTM(out_dim, dropout=dropoutrate), name='LSTM'),
        layer_num=lay_num,
        reg_index=[1, 4],
        reg_slice=(slice(None, None), slice(128, 96)),
        reg_factor=0.1,
        name='Multi-Head-Attention2',
    )(decoder_emb)
    MHC_output1 = keras.layers.Flatten(name='Flatten2')(decoder_output1)
    
    conc = layers.concatenate([peptide_output1, MHC_output1], axis=-1)
    out = Dense(2, activation='softmax')(conc)
    
    model = Model([encoder_input, decoder_input], out)  # ??????????????????????????????Model??????????????????????????????
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', tp, tn, fp, fn])
    return model
    



# data pretreatment

d = [0]*100
p =  np.array(['-'])
lp = np.array(d, dtype=float)
lps = np.append(p, lp)
lpsd = pd.DataFrame(lps).T


# class II
raw_seq_al_2 = pd.read_csv(c2_train)
raw_seq_al_2 = raw_seq_al_2.sample(frac=1).reset_index(drop=True)
raw_seq_2 = raw_seq_al_2["peptide"]
raw_seq_2MHC = raw_seq_al_2["mhc_amino_acid"]
raw_seq_al_2v = pd.read_csv(c2_val)
raw_seq_al_2v = raw_seq_al_2v.sample(frac=1).reset_index(drop=True)
raw_seq_2v = raw_seq_al_2v["peptide"]
raw_seq_2MHCv = raw_seq_al_2v["mhc_amino_acid"]


# Normalization
raw_seq_2 = replace(raw_seq_2)
raw_seq_2v = replace(raw_seq_2v)
raw_seq_2MHC = replace(raw_seq_2MHC)
raw_seq_2MHCv = replace(raw_seq_2MHCv)


feature_2 = monogram(raw_seq_2)
feature_2v = monogram(raw_seq_2v)
feature_2MHC = trigram(raw_seq_2MHC)
feature_2MHCv = trigram(raw_seq_2MHCv)
                
                     
label_2 = raw_seq_al_2["bind"]
label_2v = raw_seq_al_2v["bind"]

label_2 = pd.get_dummies(label_2, sparse=True)
label_2v = pd.get_dummies(label_2v, sparse=True)        


length_2 = []

for i in range(0, 45468):
    g = len(feature_2[i])
    length_2.append(g)
    
MAX_LEN_2 = max(length_2)

train2_x = feature_2[:45468]
train2_y = label_2[:45468]
train2_x = pad_sequences(train2_x, maxlen=MAX_LEN_2)
train2_x = np.array(train2_x)
train2_y = np.array(train2_y)

test2_x = feature_2v[:9743]
test2_y = label_2v[:9743]

test2_x = pad_sequences(test2_x, maxlen=MAX_LEN_2)

test2_x = np.array(test2_x)
test2_y = np.array(test2_y)

length_MHC = []

for i in range(0, 45468):
    g = len(feature_2MHC[i])
    length_MHC.append(g)
    
MAX_LEN_MHC = max(length_MHC)

train_x_MHC = feature_2MHC[:45468]
train_x_MHC = pad_sequences(train_x_MHC, maxlen=MAX_LEN_MHC)
train_x_MHC = np.array(train_x_MHC)


test_x_MHC = feature_2MHCv[:9743]
test_x_MHC = pad_sequences(test_x_MHC, maxlen=MAX_LEN_MHC)
test_x_MHC = np.array(test_x_MHC)

# model5 dim=128, lay=5, dropout=0.6

model = bilstmc2_model(out_dim=128, dropoutrate=0.6, lay_num=5)
H = model.fit([train2_x, train_x_MHC], train2_y, validation_data=([test2_x, test_x_MHC], test2_y), batch_size=32, verbose=1, epochs=100)
model.save('{Path_of_directory}/modal_head5.hdf5')
model.save_weights('{Path_of_directory}/modal_head5_weights.h5')
H = pd.DataFrame(H.history)
H.to_csv('{Path_of_directory}/modal_head5.csv', sep=",")

# model16 dim=512, lay=4, dropout=0.7

model = bilstmc2_model(out_dim=512, dropoutrate=0.7, lay_num=4)
H = model.fit([train2_x, train_x_MHC], train2_y, validation_data=([test2_x, test_x_MHC], test2_y), batch_size=32, verbose=1, epochs=100)
model.save('{Path_of_directory}/modal_head5.hdf16')
model.save_weights('{Path_of_directory}/modal_head16_weights.h5')
H = pd.DataFrame(H.history)
H.to_csv('{Path_of_directory}/modal_head16.csv', sep=",")






