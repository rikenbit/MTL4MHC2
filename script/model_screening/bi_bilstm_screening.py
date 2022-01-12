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

    
def multimodal_bilstm(out_dim, dropoutrate, out_dim2):
    pep_input = Input(shape=(None,)) 
    mhc_input = Input(shape=(None,)) 
    
    pep_emb = Embedding(47, 100, weights=[monovec], trainable=False)(pep_inpu)
    mhc_emb = Embedding(9419, 100, weights=[trivec1_MHC], trainable=False)(mhc_input)
    
    # peptide
    pep_output1 = Bidirectional(LSTM(out_dim, dropout=dropoutrate), merge_mode='concat')(pep_emb)
    pep_output2 = Dense(64, activation='relu')(pep_output1)
    
    # mhc
    mhc_output1 = Bidirectional(LSTM(out_dim2, dropout=dropoutrate), merge_mode='concat')(mhc_emb)
    mhc_output2 = Dense(64, activation='relu')(mhc_output1)
    
    conc = layers.concatenate([pep_output2, mhc_output2], axis=-1)
    out = Dense(2, activation='softmax')(conc)
    
    model = Model([pep_input, mhc_input], out)  
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model
    
    


# data pretreatment

d = [0]*100
p =  np.array(['-'])
lp = np.array(d, dtype=float)
lps = np.append(p, lp)
lpsd = pd.DataFrame(lps).T


# class II neoantigen
raw_seq_al_2 = pd.read_csv(c2_train)
raw_seq_al_2 = raw_seq_al_2.sample(frac=1).reset_index(drop=True)
raw_seq_2 = raw_seq_al_2["peptide"]
raw_seq_al_2v = pd.read_csv(c2_val)
raw_seq_al_2v = raw_seq_al_2v.sample(frac=1).reset_index(drop=True)
raw_seq_2v = raw_seq_al_2v["peptide"]

# class II MHC
raw_seq_2MHC = raw_seq_al_2["mhc_amino_acid"]
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

train2_x1 = feature_2[:9093]
train2_x2 = feature_2[9094:18187]
train2_x3 = feature_2[18188:27280]
train2_x4 = feature_2[27281:36374]
train2_x5 = feature_2[36375:45468]
train2_y1 = label_2[:9093]
train2_y2 = label_2[9094:18187]
train2_y3 = label_2[18188:27280]
train2_y4 = label_2[27281:36374]
train2_y5 = label_2[36375:45468]

train2_x1 = pad_sequences(train2_x1, maxlen=MAX_LEN_2)
train2_x2 = pad_sequences(train2_x2, maxlen=MAX_LEN_2)
train2_x3 = pad_sequences(train2_x3, maxlen=MAX_LEN_2)
train2_x4 = pad_sequences(train2_x4, maxlen=MAX_LEN_2)
train2_x5 = pad_sequences(train2_x5, maxlen=MAX_LEN_2)


train2_x1 = np.array(train2_x1)
train2_x2 = np.array(train2_x2)
train2_x3 = np.array(train2_x3)
train2_x4 = np.array(train2_x4)
train2_x5 = np.array(train2_x5)
train2_y1 = np.array(train2_y1)
train2_y2 = np.array(train2_y2)
train2_y3 = np.array(train2_y3)
train2_y4 = np.array(train2_y4)
train2_y5 = np.array(train2_y5)



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

train_x_MHC1 = feature_2MHC[:9093]
train_x_MHC2 = feature_2MHC[9094:18187]
train_x_MHC3 = feature_2MHC[18188:27280]
train_x_MHC4 = feature_2MHC[27281:36374]
train_x_MHC5 = feature_2MHC[36375:45468]

train_x_MHC1 = pad_sequences(train_x_MHC1, maxlen=MAX_LEN_MHC)
train_x_MHC2 = pad_sequences(train_x_MHC2, maxlen=MAX_LEN_MHC)
train_x_MHC3 = pad_sequences(train_x_MHC3, maxlen=MAX_LEN_MHC)
train_x_MHC4 = pad_sequences(train_x_MHC4, maxlen=MAX_LEN_MHC)
train_x_MHC5 = pad_sequences(train_x_MHC5, maxlen=MAX_LEN_MHC)

train_x_MHC1 = np.array(train_x_MHC1)
train_x_MHC2 = np.array(train_x_MHC2)
train_x_MHC3 = np.array(train_x_MHC3)
train_x_MHC4 = np.array(train_x_MHC4)
train_x_MHC5 = np.array(train_x_MHC5)

test_x_MHC = feature_2MHCv[:9743]
test_x_MHC = pad_sequences(test_x_MHC, maxlen=MAX_LEN_MHC)
test_x_MHC = np.array(test_x_MHC)

# parameter

out_dim = [64, 128, 256, 512]
out_dim2 = [64, 128]
dropoutrate = [0.6, 0.7]
tnx = [train2_x1, train2_x2, train2_x3, train2_x4, train2_x5]
tny = [train2_y1, train2_y2, train2_y3, train2_y4, train2_y5]
tmx = [train_x_MHC1, train_x_MHC2, train_x_MHC3, train_x_MHC4, train_x_MHC5]


for j in range(0, 5):
    for i in out_dim:
        for k in out_dim2:
            for l in dropoutrate:
                model = multimodal_bilstm(out_dim=i, dropoutrate=l, out_dim2=k)
                H = model.fit([tnx[j], tmx[j]], tny[j], validation_data=([test2_x, test_x_MHC], test2_y), batch_size=32, verbose=1, epochs=100)
                d_today = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
                H = pd.DataFrame(H.history)
                H.to_csv('{Path_of_directory}'+d_today+'.csv', sep=",")







