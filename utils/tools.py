import numpy as np
import os
import io
import math
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import pickle as pkl
from keras import backend as K

def power_spectrum(signals, num = 64653):
  all_data = []
  for signal in signals:
    fhat = np.fft.fft(signal, num) 
    PSD = fhat * np.conj(fhat) 
    out = np.expand_dims(abs(PSD), axis=-1)

    scaler = MinMaxScaler()   
    out_scaled = scaler.fit_transform(out)
    all_data.append(out_scaled)
  return np.array(all_data)


def scaler_transform(signals):
  data = []
  scale = MinMaxScaler()
  for signal in signals:
    if len(signal.shape) < 2:
      signal = np.expand_dims(signal, axis=-1)
    data.append(scale.fit_transform(signal))
  return np.array(data)

# load data from start time to end time in each audio file
def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]

# label: [normal, crackle, wheeze, both] == [0, 1, 2, 3]
def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3

# read .txt file to get annotations
def get_annotations(file_name, data_dir):
  # file_name: .txt file
  f = open(os.path.join(data_dir, file_name), "r")
  f1 = [i[:-1] for i in list(f)]
  annotations = np.array([i.split('\t') for i in f1], dtype=np.float32)
  f.close()
  return annotations

# label consists of its data
# for example: ['0': [...], '1': [...], '2': [...], '3': [...]]
# data of each label is in ...
def get_sound_samples(labels_data, annotations, file_name, data_dir, sample_rate=4000):
    # annotations: - type: list
    #              - construction: Each row of annotations is [..., ..., ..., ...] == [start, end, crackles, wheezes]
    sample_data = [file_name]
    
    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_dir, file_name), sr=sample_rate)

    for row in annotations:
        # get annotations informations
        start = row[0]
        end = row[1]
        crackles = row[2]
        wheezes = row[3]
        
        audio_chunk = slice_data(start, end, data, rate) # get segment data based on start and end time
        label_name = get_label(crackles, wheezes) # get label
        labels_data[label_name].append(audio_chunk) # put data belong to its label
    return labels_data

# load .pkz file
def load_df(pkz_file):
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

# save .pkz file
def save_df(df, out_file):
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile, protocol=4)
    print('{0} saved'.format(out_file))

# convert labels to one-hot type
def to_onehot(x, num=4):
    a = np.zeros((num, ))
    a[x] = 1
    return a.tolist()

# Convert 1D-raw data to image by stft
# w, h: width, height
# width = fft_length/2 = 224
def create_stft(current_window, frame_length=255, frame_step=100, fft_length=224*2): # increase hop -> decrease height of image
    S = tf.signal.stft(current_window, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    w, h = S.shape
    frame_step = frame_step
    while h > w:
      frame_step -= 2 # change frame_step to change height of image
      S = tf.signal.stft(current_window, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
      w, h = S.shape

    frame_step += 2
    S = tf.signal.stft(current_window, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    S = librosa.power_to_db(S, ref=np.max)
    w, h = S.shape
    img = (S-S.min()) / (S.max() - S.min()) # scale image to range of (0, 1) 

    if w < h:  # Padding zeros if height < width
      need = h-w
      l = need//2
      img_zer = np.zeros((h, h))
      img_zer[l: l+w, :] = S
      img = img_zer

    img = img[:fft_length//2, :fft_length//2]
    img = np.expand_dims(img, axis=-1) # add depth dimension
    img = np.expand_dims(img, axis=0) # add first dimention
    return img # shape: (1, w, h, 1)

# Convert 1D-raw data to image by mel spectrogram
# w, h: width, height
# width = n_mels = 224
def create_spectrograms_raw(current_window, sample_rate=4000, n_mels=224, f_min=50, f_max=4000, nfft=2048, hop=6): # increase hop -> decrease height of image
    current_window = np.array(current_window)
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    w, h = S.shape
    while h > w:
      hop += 2 # change hop to change height of image
      S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
      w, h = S.shape
    S = librosa.power_to_db(S, ref=np.max)
    w, h = S.shape
    img = (S-S.min()) / (S.max() - S.min()) # scale image to range of (0, 1) 

    if h < w:  # Padding zeros if height < width
      need = w-h
      l = need//2
      img_zer = np.zeros((w, w))
      img_zer[:, l: l+h] = S
      img = img_zer

    img = np.expand_dims(img, axis=-1) # add depth dimension
    img = np.expand_dims(img, axis=0) # add first dimention
    return img # shape: (1, w, h, 1)

############################################################ VALIDATION MATRICES #################################################
# read ICBHI_data_paper.pdf to understand matrices
def accuracy_m(y_true, y_pred):
  correct = 0
  total = 0
  for i in range(len(y_true)):
      act_label = np.argmax(y_true[i]) # act_label = 1 (index)
      pred_label = np.argmax(y_pred[i]) # pred_label = 1 (index)
      if(act_label == pred_label):
          correct += 1
      total += 1
  accuracy = (correct/total)
  return accuracy

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def sensitivity(y_true, y_pred, test=False):
  y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)
  y_true = tf.cast(tf.math.argmax(y_true, axis=-1), dtype=tf.float32)

  numerator = 0.
  denominator = 0.
  idx = 0
  for i in y_true:
    if i != 0.:
      numerator += tf.cast(y_true[idx]==y_pred[idx], tf.float32)
    idx += 1

  numerator = tf.cast(numerator, tf.float32)
  if tf.where(y_true!=0.).shape[1] == None:
    return 0.
  if test:
    denominator = tf.cast(tf.squeeze(tf.where(y_true!=0.)).shape[0], tf.float32)
  else:
    if tf.where(y_true!=0.).shape[1]:
      denominator = tf.cast(tf.where(y_true!=0.).shape[1], tf.float32)
  return numerator/denominator

def specificity(y_true, y_pred, test=False):
  y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)
  y_true = tf.cast(tf.math.argmax(y_true, axis=-1), dtype=tf.float32)

  numerator = 0.
  denominator = 0.
  idx = 0
  for i in y_true:
    if i == 0.:
      numerator += tf.cast(y_true[idx]==y_pred[idx], tf.float32)
    idx += 1

  numerator = tf.cast(numerator, tf.float32)
  if tf.where(y_true==0.).shape[1] == None:
    return 0.
  if test:
    denominator = tf.cast(tf.squeeze(tf.where(y_true==0.)).shape[0], tf.float32)
  else:
    if tf.where(y_true==0.).shape[1]:
      denominator = tf.cast(tf.where(y_true==0.).shape[1], tf.float32)
  return numerator/denominator

def average_score(y_true, y_pred, test=False):
  se = sensitivity(y_true, y_pred, test=test)
  sp = specificity(y_true, y_pred, test=test)
  return (se + sp)/2

def harmonic_mean(y_true, y_pred, test=False):
  se = sensitivity(y_true, y_pred, test=test)
  sp = specificity(y_true, y_pred, test=test)
  if se + sp == 0.:
    return 0.
  return (2*se*sp)/(se + sp)

def matrices(y_true, y_pred):
    SE = sensitivity(y_true, y_pred, True)
    SP = specificity(y_true, y_pred, True)
    AS = average_score(y_true, y_pred, True)
    HS = harmonic_mean(y_true, y_pred, True)
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.float32)
    y_true = tf.cast(tf.math.argmax(y_true, axis=-1), dtype=tf.float32)
    acc = accuracy_score(y_true, y_pred)
    return acc, SE.numpy(), SP.numpy(), AS.numpy(), HS.numpy()

##################################### MIXUP ###############################################################
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=[0.15, 0.2, 0.175], batch_size_range=[1024*5, 1024*5, 1024*5]):
    # Unpack two datasets
    images_one, ffts_one, labels_one = ds_one 
    images_two, ffts_two, labels_two = ds_two
    list_rand = list(range(len(labels_one)))

    # Random the first data
    np.random.shuffle(list_rand)
    images_one, ffts_one, labels_one = images_one[list_rand], ffts_one[list_rand], labels_one[list_rand]

    # Random the second data
    np.random.shuffle(list_rand)
    images_two, ffts_two, labels_two = images_two[list_rand], ffts_two[list_rand], labels_two[list_rand]
    images_org, ffts_org, labels_org = ds_one 

    for idx, batch_size in enumerate(batch_size_range):
      num = int(len(labels_one)/batch_size)
      for i in range(num):
        # Sample lambda and reshape it to do the mixup
        l = sample_beta_distribution(batch_size, alpha[idx], alpha[idx])
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        x_f = tf.reshape(l, (batch_size, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images_one_batch = images_one[i*batch_size: (i+1)*batch_size]
        images_two_batch = images_two[i*batch_size: (i+1)*batch_size]

        ffts_one_batch = ffts_one[i*batch_size: (i+1)*batch_size]
        ffts_two_batch = ffts_two[i*batch_size: (i+1)*batch_size]

        labels_one_batch = labels_one[i*batch_size: (i+1)*batch_size]
        labels_two_batch = labels_two[i*batch_size: (i+1)*batch_size]

        images = images_one_batch*x_l + images_two_batch*(1 - x_l)
        ffts   = ffts_one_batch  *x_f + ffts_two_batch  *(1 - x_f)
        labels = labels_one_batch*y_l + labels_two_batch*(1 - y_l)

        images_org = np.concatenate((images_org, images), axis=0)
        ffts_org = np.concatenate((ffts_org, ffts), axis=0)
        labels_org = np.concatenate((labels_org, labels), axis=0)
    return images_org, ffts_org, labels_org

def convert_fft(x, n=64653):
  data_fft = []
  for i in x:
    t = np.fft.fft(i, n=n)
    data_fft.append(abs(t))
  return np.expand_dims(data_fft, -1)

def scaler_signal(signals):
  data = []
  scaler = MinMaxScaler()   
  for signal in signals:
    sig = scaler.fit_transform(signal.reshape(-1, 1))
    sig = sig.reshape(-1, )
    data.append(sig)
  return data

def arrange_data(signals, num=64653):
  data = []
  signals = scaler_signal(signals)
  for signal in signals:
    length = signal.shape[0]
    if length < num:
      signal = np.concatenate((signal, np.zeros(num - length, dtype=float)))
    if length > num:
      signal = signal[:num]
    data.append(signal.tolist())
  data = np.array(data)
  return data
