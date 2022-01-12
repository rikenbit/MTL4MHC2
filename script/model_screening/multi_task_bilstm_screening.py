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


def MTL_bilstm(out_dim1, dropoutrate, loss1, loss2):
    shared_embedding = Embedding(47, 100, weights=[monovec], trainable=False)
    sharedLSTM1 = LSTM(out_dim1, dropout=dropoutrate, return_sequences=True)
    sharedLSTM2 = LSTM(out_dim1,dropout=dropoutrate)
    sharedLSTM_bw1 = LSTM(out_dim1, dropout=dropoutrate, return_sequences=True, go_backwards=True)
    sharedLSTM_bw2 = LSTM(out_dim1, dropout=dropoutrate, go_backwards=True)
    
    shared_embedding_MHC = Embedding(9419, 100, input_length=230, weights=[trivec1_MHC], trainable=False)
    sharedLSTM1_MHC = LSTM(out_dim1, dropout=dropoutrate, return_sequences=True)
    sharedLSTM2_MHC = LSTM(out_dim1,dropout=dropoutrate)
    sharedLSTM_bw1_MHC = LSTM(out_dim1, dropout=dropoutrate, return_sequences=True, go_backwards=True)
    sharedLSTM_bw2_MHC = LSTM(out_dim1, dropout=dropoutrate, go_backwards=True)
    
    
    
    text_input_c1 = keras.Input(shape=(None,))
    text_input_c2 = keras.Input(shape=(None,))
    
    encoded_input_c1 = shared_embedding(text_input_c1)
    encoded_input_c2 = shared_embedding(text_input_c2)
    
    #class I
    sharedLSTM1Instance_c1 = sharedLSTM1(encoded_input_c1)
    sharedLSTM_bw1Instance_c1 = sharedLSTM_bw1(encoded_input_c1)
    BiLSTM_c1_output_1 = layers.concatenate([sharedLSTM1Instance_c1, sharedLSTM_bw1Instance_c1], axis=-1)
    sharedLSTM2Instance_c1 =  sharedLSTM2(BiLSTM_c1_output_1)
    sharedLSTM_bw2Instance_c1 =  sharedLSTM_bw2(BiLSTM_c1_output_1)
    BiLSTM_c1_output_2 = layers.concatenate([sharedLSTM2Instance_c1, sharedLSTM_bw2Instance_c1])
    BiLSTM_c1_output_3 = Dense(64, activation='relu')(BiLSTM_c1_output_2)



    
    # class II
    sharedLSTM1Instance_c2 = sharedLSTM1(encoded_input_c2)
    sharedLSTM_bw1Instance_c2 = sharedLSTM_bw1(encoded_input_c2)
    BiLSTM_c2_output_1 = layers.concatenate([sharedLSTM1Instance_c2, sharedLSTM_bw1Instance_c2], axis=-1)
    sharedLSTM2Instance_c2 =  sharedLSTM2(BiLSTM_c2_output_1)
    sharedLSTM_bw2Instance_c2 =  sharedLSTM_bw2(BiLSTM_c2_output_1)
    BiLSTM_c2_output_2 = layers.concatenate([sharedLSTM2Instance_c2, sharedLSTM_bw2Instance_c2])
    BiLSTM_c2_output_3 = Dense(64, activation='relu')(BiLSTM_c2_output_2)


    
    text_input_c1_MHC = keras.Input(shape=(None,))
    text_input_c2_MHC = keras.Input(shape=(None,))
    
    encoded_input_c1_MHC = shared_embedding_MHC(text_input_c1_MHC)
    encoded_input_c2_MHC = shared_embedding_MHC(text_input_c2_MHC)
    
    # class I
    sharedLSTM1Instance_c1_MHC = sharedLSTM1_MHC(encoded_input_c1_MHC)
    sharedLSTM_bw1Instance_c1_MHC = sharedLSTM_bw1_MHC(encoded_input_c1_MHC)
    BiLSTM_c1_output_1_MHC = layers.concatenate([sharedLSTM1Instance_c1_MHC, sharedLSTM_bw1Instance_c1_MHC], axis=-1)
    sharedLSTM2Instance_c1_MHC =  sharedLSTM2_MHC(BiLSTM_c1_output_1_MHC)
    sharedLSTM_bw2Instance_c1_MHC =  sharedLSTM_bw2_MHC(BiLSTM_c1_output_1_MHC)
    BiLSTM_c1_output_2_MHC = layers.concatenate([sharedLSTM2Instance_c1_MHC, sharedLSTM_bw2Instance_c1_MHC])
    BiLSTM_c1_output_3_MHC = Dense(64, activation='relu')(BiLSTM_c1_output_2_MHC)


    
    # class II
    sharedLSTM1Instance_c2_MHC = sharedLSTM1_MHC(encoded_input_c2_MHC)
    sharedLSTM_bw1Instance_c2_MHC = sharedLSTM_bw1_MHC(encoded_input_c2_MHC)
    BiLSTM_c2_output_1_MHC = layers.concatenate([sharedLSTM1Instance_c2_MHC, sharedLSTM_bw1Instance_c2_MHC], axis=-1)
    sharedLSTM2Instance_c2_MHC =  sharedLSTM2_MHC(BiLSTM_c2_output_1_MHC)
    sharedLSTM_bw2Instance_c2_MHC =  sharedLSTM_bw2_MHC(BiLSTM_c2_output_1_MHC)
    BiLSTM_c2_output_2_MHC = layers.concatenate([sharedLSTM2Instance_c2_MHC, sharedLSTM_bw2Instance_c2_MHC])
    BiLSTM_c2_output_3_MHC = Dense(64, activation='relu')(BiLSTM_c2_output_2_MHC)
    
    
    conc1 = layers.concatenate([BiLSTM_c1_output_3, BiLSTM_c1_output_3_MHC], axis=-1)
    conc2 = layers.concatenate([BiLSTM_c2_output_3, BiLSTM_c2_output_3_MHC], axis=-1)
    out1 = Dense(2, activation='softmax')(conc1)
    out2 = Dense(2, activation='softmax')(conc2)
    
    
    model = Model([text_input_c1, text_input_c2, text_input_c1_MHC, text_input_c2_MHC], outputs=[out1, out2])  
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy',], loss_weights=[loss1, loss2], optimizer="adam", metrics=['accuracy'])
    
    
    
    return model



# data pretreatment

d = [0]*100
p =  np.array(['-'])
lp = np.array(d, dtype=float)
lps = np.append(p, lp)
lpsd = pd.DataFrame(lps).T


# class I
raw_seq_al_1 = pd.read_csv(c1_train)
raw_seq_al_1 = raw_seq_al_1.sample(frac=1).reset_index(drop=True)
raw_seq_1 = raw_seq_al_1["peptide"]
raw_seq_1MHC = raw_seq_al_1["mhc_amino_acid"]
raw_seq_al_1v = pd.read_csv(c1_val)
raw_seq_al_1v = raw_seq_al_1v.sample(frac=1).reset_index(drop=True)
raw_seq_1v = raw_seq_al_1v["peptide"]
raw_seq_1MHCv = raw_seq_al_1v["mhc_amino_acid"]


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
raw_seq_1 = replace(raw_seq_1)
raw_seq_1v = replace(raw_seq_1v)
raw_seq_2 = replace(raw_seq_2)
raw_seq_2v = replace(raw_seq_2v)
raw_seq_1MHC = replace(raw_seq_1MHC)
raw_seq_1MHCv = replace(raw_seq_1MHCv)
raw_seq_2MHC = replace(raw_seq_2MHC)
raw_seq_2MHCv = replace(raw_seq_2MHCv)

   
feature_1 = monogram(raw_seq_1)
feature_1v = monogram(raw_seq_1v)
feature_2 = monogram(raw_seq_2)
feature_2v = monogram(raw_seq_2v)

feature_1MHC = trigram(raw_seq_1MHC)
feature_1MHCv = trigram(raw_seq_1MHCv)
feature_2MHC = trigram(raw_seq_2MHC)
feature_2MHCv = trigram(raw_seq_2MHCv)

                     
label_1 = raw_seq_al_1["bind"]
label_1v = raw_seq_al_1v["bind"]

label_1 = pd.get_dummies(label_1, sparse=True)
label_1v = pd.get_dummies(label_1v, sparse=True)                
                     
label_2 = raw_seq_al_2["bind"]
label_2v = raw_seq_al_2v["bind"]

label_2 = pd.get_dummies(label_2, sparse=True)
label_2v = pd.get_dummies(label_2v, sparse=True)   


length_1 = []

for i in range(0, 343713):
    g = len(feature_1[i])
    length_1.append(g)
    
MAX_LEN_1 = max(length_1)

# varidation data

train1_x1 = feature_1[:9093]
train1_x2 = feature_1[9094:18187]
train1_x3 = feature_1[18188:27280]
train1_x4 = feature_1[27281:36374]
train1_x5 = feature_1[36375:45468]
train1_y1 = label_1[:9093]
train1_y2 = label_1[9094:18187]
train1_y3 = label_1[18188:27280]
train1_y4 = label_1[27281:36374]
train1_y5 = label_1[36375:45468]

train1_x1 = pad_sequences(train1_x1, maxlen=MAX_LEN_1)
train1_x2 = pad_sequences(train1_x2, maxlen=MAX_LEN_1)
train1_x3 = pad_sequences(train1_x3, maxlen=MAX_LEN_1)
train1_x4 = pad_sequences(train1_x4, maxlen=MAX_LEN_1)
train1_x5 = pad_sequences(train1_x5, maxlen=MAX_LEN_1)


train1_x1 = np.array(train1_x1)
train1_x2 = np.array(train1_x2)
train1_x3 = np.array(train1_x3)
train1_x4 = np.array(train1_x4)
train1_x5 = np.array(train1_x5)
train1_y1 = np.array(train1_y1)
train1_y2 = np.array(train1_y2)
train1_y3 = np.array(train1_y3)
train1_y4 = np.array(train1_y4)
train1_y5 = np.array(train1_y5)



test1_x = feature_1v[:9743]
test1_y = label_1v[:9743]

test1_x = pad_sequences(test1_x, maxlen=MAX_LEN_1)

test1_x = np.array(test1_x)
test1_y = np.array(test1_y)



length_1MHC = []

for i in range(0, 343713):
    g = len(feature_1MHC[i])
    length_1MHC.append(g)
    
MAX_LEN_1MHC = max(length_1MHC)

#

train1_x1MHC = feature_1MHC[:9093]
train1_x2MHC = feature_1MHC[9094:18187]
train1_x3MHC = feature_1MHC[18188:27280]
train1_x4MHC = feature_1MHC[27281:36374]
train1_x5MHC = feature_1MHC[36375:45468]

train1_x1MHC = pad_sequences(train1_x1MHC, maxlen=MAX_LEN_1MHC)
train1_x2MHC = pad_sequences(train1_x2MHC, maxlen=MAX_LEN_1MHC)
train1_x3MHC = pad_sequences(train1_x3MHC, maxlen=MAX_LEN_1MHC)
train1_x4MHC = pad_sequences(train1_x4MHC, maxlen=MAX_LEN_1MHC)
train1_x5MHC = pad_sequences(train1_x5MHC, maxlen=MAX_LEN_1MHC)


train1_x1MHC = np.array(train1_x1MHC)
train1_x2MHC = np.array(train1_x2MHC)
train1_x3MHC = np.array(train1_x3MHC)
train1_x4MHC = np.array(train1_x4MHC)
train1_x5MHC = np.array(train1_x5MHC)

train1_xMHC = feature_1MHC[:45468]
train1_xMHC = pad_sequences(train1_xMHC, maxlen=MAX_LEN_1MHC)
train1_xMHC = np.array(train1_xMHC)

test1_xMHC = feature_1MHCv[:9743]
test1_xMHC = pad_sequences(test1_xMHC, maxlen=MAX_LEN_1MHC)
test1_xMHC = np.array(test1_xMHC)




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

train2_x1 = pad_sequences(train2_x1, maxlen=MAX_LEN_1)
train2_x2 = pad_sequences(train2_x2, maxlen=MAX_LEN_1)
train2_x3 = pad_sequences(train2_x3, maxlen=MAX_LEN_1)
train2_x4 = pad_sequences(train2_x4, maxlen=MAX_LEN_1)
train2_x5 = pad_sequences(train2_x5, maxlen=MAX_LEN_1)


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

test2_x = pad_sequences(test2_x, maxlen=MAX_LEN_1)

test2_x = np.array(test2_x)
test2_y = np.array(test2_y)



train2_x1MHC = feature_2MHC[:9093]
train2_x2MHC = feature_2MHC[9094:18187]
train2_x3MHC = feature_2MHC[18188:27280]
train2_x4MHC = feature_2MHC[27281:36374]
train2_x5MHC = feature_2MHC[36375:45468]

train2_x1MHC = pad_sequences(train2_x1MHC, maxlen=MAX_LEN_1MHC)
train2_x2MHC = pad_sequences(train2_x2MHC, maxlen=MAX_LEN_1MHC)
train2_x3MHC = pad_sequences(train2_x3MHC, maxlen=MAX_LEN_1MHC)
train2_x4MHC = pad_sequences(train2_x4MHC, maxlen=MAX_LEN_1MHC)
train2_x5MHC = pad_sequences(train2_x5MHC, maxlen=MAX_LEN_1MHC)


train2_x1MHC = np.array(train2_x1MHC)
train2_x2MHC = np.array(train2_x2MHC)
train2_x3MHC = np.array(train2_x3MHC)
train2_x4MHC = np.array(train2_x4MHC)
train2_x5MHC = np.array(train2_x5MHC)

test2_xMHC = feature_2MHCv[:9743]
test2_xMHC = pad_sequences(test2_xMHC, maxlen=MAX_LEN_1MHC)
test2_xMHC = np.array(test2_xMHC)



#parameter

out_dim1 = [128, 256, 512]
dropoutrate = [0.6, 0.7]
tnx = [train1_x1, train1_x2, train1_x3, train1_x4, train1_x5]
tny = [train1_y1, train1_y2, train1_y3, train1_y4, train1_y5]
tmx = [train2_x1, train2_x2, train2_x3, train2_x4, train2_x5]
tmy = [train2_y1, train2_y2, train2_y3, train2_y4, train2_y5]
tnx_MHC = [train1_x1MHC, train1_x2MHC, train1_x3MHC, train1_x4MHC, train1_x5MHC]
tmx_MHC = [train2_x1MHC, train2_x2MHC, train2_x3MHC, train2_x4MHC, train2_x5MHC]



for j in range(0, 5):
    for i in out_dim1:
        for l in dropoutrate:
            model = MTL_bilstm(out_dim1=i, out_dim3=k, dropoutrate=l, loss1=10, loss2=90)
            H = model.fit([tnx[j], tmx[j],tnx_MHC[j], tmx_MHC[j] ], [tny[j], tmy[j]], validation_data=([test1_x, test2_x, test1_xMHC, test2_xMHC], [test1_y, test2_y]), batch_size=32, verbose=1, epochs=100)
            d_today = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            H = pd.DataFrame(H.history)
            H.to_csv('{Path_of_save_directory}'+d_today+'.csv', sep=",")





