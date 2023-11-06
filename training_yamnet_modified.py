# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:12:23 2022

@author: Fahimeh
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

yamnet_base = 'C:/Users/Asus/models/research/audioset/yamnet/'
sys.path.append(yamnet_base)

## Yamnet imports 
import params as yamnet_params
import yamnet  as yamnet_model
import features as features_lib

chkp=True

params = yamnet_params.Params()

#creat the same csv file for your costum dataset 
data='C:/Users/Asus/models/research/audioset/yamnet/yamnet_class_map_1.csv'
df=pd.read_csv(data)

class_names = yamnet_model.class_names('C:/Users/Asus/models/research/audioset/yamnet/yamnet_class_map_1.csv')

# define the yamnet model. The yamnet_frames_model_transfer function is
#modified to use on the custom data set





#read the wav files
@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    
    return wav
 
def load_wav_for_map(filename, label,fold):
  return load_wav_16k_mono(filename), label,fold
   
#open your custom dataset

data1='D:/bat_n/df_train.csv'
df_train=pd.read_csv(data1)


df_train.drop(['category'],inplace=True,axis=1)

#folds is used to split the dataset to train val and test
filenames=df_train['filenames']
targets=df_train['targets']
folds=df_train['folds']



# This function recived the wav file and each wav file divid to frames with 
#96ms longe and 10ms hope.for each frame the lable is the label of the
#main audio file.Then a batch of these frames is used as input to the yamnet model
def yamnet_frames_model_transfer1(wav_data,label,fold):
   
    waveform_padded = features_lib.pad_waveform(wav_data, params)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
        waveform_padded, params)
    num_embeddings = tf.shape(features)[0]
    print(log_mel_spectrogram.shape)
    
    
    return log_mel_spectrogram, label,fold


#creat the dataset as tensor
main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets,folds))

#main_ds.element_spec
#apply the load_wav_for_map on the dataset. by thos map still no data is loaded on the memory 
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


#divid the wav files to the frames.  unbatch() is very importatnt.
#for given wav file the size of the yamnet_frames_model_transfer1 function output is 
#(m,96,64). when using .unbatch() it gives m tensor with (96,64) size.
# a batch of these array with size (96,64) is input of the model
main_ds = main_ds.map(yamnet_frames_model_transfer1)
#The last step is to remove the fold column from the dataset since
# you're not going to use it during training.
cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, folds: folds <=3)
val_ds = cached_ds.filter(lambda embedding, label, folds: folds <=4)
#test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)

val_ds = val_ds.map(remove_fold_column)

#test_ds = test_ds.map(remove_fold_column)


#X_train = list(map(lambda x: x[0], train_ds))
#y_train = list(map(lambda x: x[1], train_ds))

#creat a batch of size 32 of frames with size (96,64)
#we have to suffle the train set to avoid the frames from the same audio on one batch
train_ds = train_ds.cache().shuffle(1000).batch(32*2).prefetch( tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().batch(32*2).prefetch( tf.data.experimental.AUTOTUNE)



#train_ds=train_ds.map(one_hot_label)
#val_ds=val_ds.map(one_hot_label)
'''
#by this for we can see the size of the input
for element in main_ds.as_numpy_iterator(): 
    
  print('element=',element[0].shape) 
  
'''





#load yamnet model. yamnet_frames_model_transfer1 is modified version of the
#yamnet_frames_model_transfer in yamnet.py file in order to be able the 
#train yamnet from scratch


yamnet=yamnet_model.yamnet_frames_model_transfer(params,1)

preloaded_layers = yamnet.layers.copy()
preloaded_weights = []
for pre in preloaded_layers:
        preloaded_weights.append(pre.get_weights())    


#load the weights from pretrain model except for the last layer and
#check which layer used the pretrain weights
# store weights before loading pre-trained weights

if chkp==True:
# load pre-trained weights(fine tuning the model)
#load the weights from pretrain model except for the last layer
    yamnet.load_weights('C:/Users/Asus/models/research/audioset/yamnet/yamnet.h5',by_name=True)
 #   yamnet.load_weights('D:/bat_n/yamnet_2.h5',by_name=True)
    for layer, pre in zip(yamnet.layers, preloaded_weights):
        weights = layer.get_weights()
        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)

NAME='D:/bat_n/yamnet_2.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

tensorboard=tf.keras.callbacks.TensorBoard(
    log_dir='D:/bat_n/logs')

yamnet.compile(optimizer='adam', 
               loss="BinaryCrossentropy", 
               metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
yamnet.fit(train_ds,epochs=100, validation_data= val_ds,callbacks=[checkpoint,tensorboard,callback])






loss=yamnet.evaluate(test_ds)

#test the model

#dir_="D:/bat_n/df_test_b.csv"
#dir_="D:/bat_n/df_test_n.csv"
#dir_="D:/bat_n/df_test_uk.csv"
import os
dir_="D:/bat_n/norfolk_test_files.csv"
df_test_b=pd.read_csv(dir_)
base_data_path='D:/bat_n/wav/'
full_path = df_test_b['filename'].apply(lambda row: os.path.join(base_data_path, row))
df_test_b= df_test_b.assign(filename=full_path)

full_path = df_test_b['filename'].apply(lambda row: ( row+ '.wav'))

df_test_b= df_test_b.assign(filename=full_path)

filenames=df_test_b['filename']
targets=df_test_b['target']
df_test_b['fold']=1
folds=df_test_b['fold']


#the directory contained the .wav files

test_b = tf.data.Dataset.from_tensor_slices((filenames, targets,folds))
test_b= test_b.map(load_wav_for_map)
test_b = test_b.map(yamnet_frames_model_transfer1).unbatch()
remove_fold_column = lambda embedding, label, fold: (embedding, label)
test_b = test_b.map(remove_fold_column)
test_b = test_b.cache().batch(32).prefetch( tf.data.experimental.AUTOTUNE)

evaluate= yamnet.evaluate(test_b)







dir_="D:/bat_n/df_test_uk.csv"
dir_="D:/bat_n/uk_test_files1.csv"
df_test_b=pd.read_csv(dir_)


filenames=df_test_b['filename']
targets=df_test_b['target']
df_test_b['fold']=1
folds=df_test_b['fold']


#the directory contained the .wav files

test_b = tf.data.Dataset.from_tensor_slices((filenames, targets,folds))
test_b= test_b.map(load_wav_for_map)
test_b = test_b.map(yamnet_frames_model_transfer1).unbatch()
cached_ds = main_ds.cache()
test_b_train = cached_ds.filter(lambda embedding, label, fold: fold <2)
test_b_val = cached_ds.filter(lambda embedding, label, fold: fold ==3)
test_b_test = cached_ds.filter(lambda embedding, label, fold: fold == 4)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

test_b_train= test_b_train.map(remove_fold_column)

test_b_val= test_b_val.map(remove_fold_column)

test_b_test= test_b_test.map(remove_fold_column)


#X_train = list(map(lambda x: x[0], train_ds))
#y_train = list(map(lambda x: x[1], train_ds))

#creat a batch of size 32 of frames with size (96,64)
#we have to suffle the train set to avoid the frames from the same audio on one batch
train_ds = test_b_train.cache().shuffle(1000).batch(32).prefetch( tf.data.experimental.AUTOTUNE)
val_ds = test_b_val.cache().batch(32).prefetch( tf.data.experimental.AUTOTUNE)
test_ds = test_b_test.cache().batch(32).prefetch( tf.data.experimental.AUTOTUNE)



#test n

'''
dir_="D:/bat_n/df_test_n.csv"
dir_="D:/bat_n/df_test_uk.csv"
df_test_b=pd.read_csv(dir_)

filenames=df_test_b['filename']
targets=df_test_b['target']
folds=df_test_b['fold']
l=[]
for j in range(1,5):
    print((j-1),'--',j*175)
    for i in range(0,175):
        
        l.append(j)
        
        
        
folds=l[:len(df_test_b)]        
        
    

test_b = tf.data.Dataset.from_tensor_slices((filenames, targets,folds))
test_b= test_b.map(load_wav_for_map)

test_b = test_b.map(yamnet_frames_model_transfer1)#.unbatch()


cached_ds = test_b.cache()
test_b_train = cached_ds.filter(lambda embedding, label, fold: fold <2)
test_b_val = cached_ds.filter(lambda embedding, label, fold: fold ==3)
test_b_test = cached_ds.filter(lambda embedding, label, fold: fold <= 2)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

test_b_train= test_b_train.map(remove_fold_column)

test_b_val= test_b_val.map(remove_fold_column)

test_b_test= test_b_test.map(remove_fold_column)



#creat a batch of size 32 of frames with size (96,64)
#we have to suffle the train set to avoid the frames from the same audio on one batch
train_ds = test_b_train.cache().shuffle(1000).batch(32).prefetch( tf.data.experimental.AUTOTUNE)
val_ds = test_b_val.cache().batch(32).prefetch( tf.data.experimental.AUTOTUNE)
test_ds = test_b_test.cache().batch(32).prefetch( tf.data.experimental.AUTOTUNE)


yamnet=yamnet_model.yamnet_frames_model_transfer(params)

preloaded_layers = yamnet.layers.copy()
preloaded_weights = []
for pre in preloaded_layers:
        preloaded_weights.append(pre.get_weights())    


#load the weights from pretrain model except for the last layer and
#check which layer used the pretrain weights
# store weights before loading pre-trained weights
chkp==True
if chkp==True:
# load pre-trained weights(fine tuning the model)
#load the weights from pretrain model except for the last layer
    yamnet.load_weights('D:/bat_n/yamnet_2.h5',by_name=True)
    for layer, pre in zip(yamnet.layers, preloaded_weights):
        weights = layer.get_weights()
        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)





yamnet.compile(optimizer='adam', 
               loss='sparse_categorical_crossentropy', 
               metrics=['accuracy'])
yamnet.fit(train_ds,epochs=20)






loss= yamnet.evaluate(test_ds)





from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

SAMPLE_RATE = 16000
X=list(map(lambda x: x[0], test_b))
X=np.array(X)
#y_train = list(map(lambda x: x[1], train_ds))
augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])
augmented_samples = augment(samples=X, sample_rate=16000)
    
    
'''