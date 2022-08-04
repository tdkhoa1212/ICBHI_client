import os
import itertools
import argparse
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nets.CNN import EfficientNetV2M, MobileNetV2, InceptionResNetV2, ResNet152V2
from nets.CNN_1D_2D import CNN_1D_2D_model
from sklearn.model_selection import train_test_split
from utils.tools import to_onehot, load_df, create_spectrograms_raw, \
                        get_annotations, get_sound_samples, save_df, sensitivity, \
                        specificity, average_score, harmonic_mean, \
                        matrices, create_stft, mix_up, convert_fft
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import progressbar

# input argmuments
parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification')
parser.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
parser.add_argument('--image_length', default = 224, type=int, help='height and width of image')
parser.add_argument('--fft_length', default = 64653, type=int, help='length of frequency signal')
parser.add_argument('--batch_size', default = 16, type=int, help='bacth size')
parser.add_argument('--epochs', default = 100, type=int, help='epochs')
parser.add_argument('--load_weight', default = False, type=bool, help='load weight')
parser.add_argument('--model_name', type=str, help='names of model: EfficientNetV2M, MobileNetV2, InceptionResNetV2, ResNet152V2')

parser.add_argument('--save_data_dir', type=str, help='data directory: x/x/')
parser.add_argument('--data_dir', type=str, help='data directory: x/x/ICBHI_final_database')
parser.add_argument('--model_path', type=str, help='model saving directory')

parser.add_argument('--train', type=bool, default=False, help='train mode')
parser.add_argument('--predict', type=bool, default=False, help='predict mode')

parser.add_argument('--based_image', type=str, default='mel', help='stft, mel')
args = parser.parse_args()

def train(args):
    ######################## LOAD DATA ##################################################################
    if os.path.exists(os.path.join(args.save_data_dir, 'test_data.pkz')):
        # if raw data was splitted before, the splitted data will be loaded data from saved files (.pkz)
        test_data = load_df(os.path.join(args.save_data_dir, 'test_data.pkz'))
        test_label = load_df(os.path.join(args.save_data_dir, 'test_label.pkz'))
        train_data = load_df(os.path.join(args.save_data_dir, 'train_data.pkz'))
        train_label = load_df(os.path.join(args.save_data_dir, 'train_label.pkz'))
    else:
        # Load file names 
        print('\n' + '-'*10 + 'CATAGORIZE DATA' + '-'*10)
        files_name = []
        for i in os.listdir(args.data_dir):
            tail = i.split('.')[-1]
            head = i.split('.')[0]
            if tail == 'wav':
                files_name.append(head)

        # label (before onehot): normal, crackles, wheezes, both = 0, 1, 2, 3
        labels_data = {0: [], 1: [], 2: [], 3: []}
        for file_name in files_name:
            audio_file = file_name + '.wav' # audio file names
            txt_file = file_name + '.txt' # annotations file names
            annotations = get_annotations(txt_file, args.data_dir) # loading annotations 
            labels_data = get_sound_samples(labels_data, annotations, audio_file, args.data_dir, sample_rate=4000) # loading labels according to the form: normal, crackles, wheezes, both = 0, 1, 2, 3
        
        # split data to test and train set.
        # In each type of label: splitting 80% for train, 20% for test following the paper.
        test_data = []
        test_label = []
        train_data = []
        train_label = []
        for name in labels_data:
            all_data = labels_data[name]
            label = [name]*len(all_data)
            all_label = np.array([to_onehot(i) for i in label]) # convert label to one-hot type
            
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=42) 
            
            # gather data for train set
            if train_data == []:
                train_data = X_train
                train_label = y_train
            else:
                train_data = np.concatenate((train_data, X_train))
                train_label = np.concatenate((train_label, y_train))
            
            # gather data for test set
            if test_data == []:
                test_data = X_test
                test_label = y_test
            else:
                test_data = np.concatenate((test_data, X_test))
                test_label = np.concatenate((test_label, y_test))
        
        # save splitted data
        save_df(test_data, os.path.join(args.save_data_dir, 'test_data.pkz'))
        save_df(test_label, os.path.join(args.save_data_dir, 'test_label.pkz'))
        save_df(train_data, os.path.join(args.save_data_dir, 'train_data.pkz'))
        save_df(train_label, os.path.join(args.save_data_dir, 'train_label.pkz'))
        print('\n' + '-'*10 + 'SAVED DATA' + '-'*10)
    
    ######################## PREPROCESSING DATA ##################################################################
    if args.based_image == 'mel': # convert raw data to mel spectrogram
        if os.path.exists(os.path.join(args.save_data_dir, 'mel_test_data.pkz')):
          # Load mel spectrogram data, if they exist
          image_test_data = load_df(os.path.join(args.save_data_dir, 'mel_test_data.pkz'))
          image_train_data = load_df(os.path.join(args.save_data_dir, 'mel_train_data.pkz'))
        else:
          image_test_data = []
          image_train_data = []
          
          # start convert 1D test data to mel spectrogram
          print('\n' + 'Convert test data: ...')
          p_te = progressbar.ProgressBar(maxval=len(test_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
          p_te.start()
          for idx_te, te in enumerate(test_data):
              p_te.update(idx_te+1)
              if len(image_test_data) == 0:
                  image_test_data = create_spectrograms_raw(te, n_mels=args.image_length) # API for convert mel spectrogram. It is in utils/tool.py
              else:
                  image_test_data = np.concatenate((image_test_data, create_spectrograms_raw(te, n_mels=args.image_length)), axis=0)
          p_te.finish()
          
          # start convert 1D train data to mel spectrogram
          print('\n' + 'Convert train data: ...')
          p_tra = progressbar.ProgressBar(maxval=len(train_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
          p_tra.start()      
          for idx_tra, tra in enumerate(train_data):
              p_tra.update(idx_tra+1)
              if len(image_train_data) == 0:
                  image_train_data = create_spectrograms_raw(tra, n_mels=args.image_length)
              else:
                  image_train_data = np.concatenate((image_train_data, create_spectrograms_raw(tra, n_mels=args.image_length)), axis=0)
          p_tra.finish()
          
          # save test and train data
          save_df(image_test_data, os.path.join(args.save_data_dir, 'mel_test_data.pkz'))
          save_df(image_train_data, os.path.join(args.save_data_dir, 'mel_train_data.pkz'))
    
    if args.based_image == 'stft':
        if os.path.exists(os.path.join(args.save_data_dir, 'stft_test_data.pkz')):
          # Load stft data, if they exist
          image_test_data = load_df(os.path.join(args.save_data_dir, 'stft_test_data.pkz'))
          image_train_data = load_df(os.path.join(args.save_data_dir, 'stft_train_data.pkz'))
        else:
          image_test_data = []
          image_train_data = []
          
          # start convert 1D test data to stft
          print('\n' + 'Convert test data: ...')
          p_te = progressbar.ProgressBar(maxval=len(test_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
          p_te.start()
          for idx_te, te in enumerate(test_data):
              p_te.update(idx_te+1)
              if len(image_test_data) == 0:
                  image_test_data = create_stft(te)
              else:
                  image_test_data = np.concatenate((image_test_data, create_stft(te)), axis=0)
          p_te.finish()
          
          # start convert 1D train data to stft
          print('\n' + 'Convert train data: ...')
          p_tra = progressbar.ProgressBar(maxval=len(train_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
          p_tra.start()      
          for idx_tra, tra in enumerate(train_data):
              p_tra.update(idx_tra+1)
              if len(image_train_data) == 0:
                  image_train_data = create_stft(tra)
              else:
                  image_train_data = np.concatenate((image_train_data, create_stft(tra)), axis=0)
          p_tra.finish()
           
          # save stft-form data
          save_df(image_test_data, os.path.join(args.save_data_dir, 'stft_test_data.pkz'))
          save_df(image_train_data, os.path.join(args.save_data_dir, 'stft_train_data.pkz'))


    ######################## TRAIN PHASE ##################################################################
    print(f'\nShape of train data: {image_train_data.shape} \t {train_label.shape}')
    print(f'Shape of test data: {image_test_data.shape} \t {test_label.shape}\n')

    image_train_data = image_train_data.astype(np.float32)
    image_test_data = image_test_data.astype(np.float32)

    train_label = train_label.astype(np.float32)
    test_label = test_label.astype(np.float32)

    # ---------------------------------Convert data to frequency by FFT------------------------------------
    if os.path.exists(os.path.join(args.save_data_dir, 'train_fft.pkz')):
      train_fft = load_df(os.path.join(args.save_data_dir, 'train_fft.pkz'))
      test_fft = load_df(os.path.join(args.save_data_dir, 'test_fft.pkz'))
    else:
      train_fft = convert_fft(train_data)
      test_fft = convert_fft(test_data)
      save_df(train_fft, os.path.join(args.save_data_dir, 'train_fft.pkz'))
      save_df(test_fft, os.path.join(args.save_data_dir, 'test_fft.pkz'))

    print(f'\nShape of 1D training data{train_fft.shape}')
    print(f'Shape of 1D test data{test_fft.shape}\n')
    
    #-------------------------- MIXUP --------------------------------------------------------------------
#     if os.path.exists(os.path.join(args.save_data_dir, 'images_org.pkz')):
#       images_org = load_df(os.path.join(args.save_data_dir, 'images_org.pkz'))
#       ffts_org = load_df(os.path.join(args.save_data_dir, 'ffts_org.pkz'))
#       labels_org = load_df(os.path.join(args.save_data_dir, 'labels_org.pkz'))
#     else:
    train_ds_one = (image_train_data, train_fft, train_label)
    train_ds_two = (image_train_data, train_fft, train_label)
    images_org, ffts_org, labels_org = mix_up(train_ds_one, train_ds_two)
#       save_df(images_org, os.path.join(args.save_data_dir, 'images_org.pkz'))
#       save_df(ffts_org, os.path.join(args.save_data_dir, 'ffts_org.pkz'))
#       save_df(labels_org, os.path.join(args.save_data_dir, 'labels_org.pkz'))


    print(f'\nShape of 1D MIXUP training data: {images_org.shape}, {ffts_org.shape}, {labels_org.shape}\n')
    # load neural network model
    if args.model_name == 'EfficientNetV2M':
      model = EfficientNetV2M(args.image_length, True)
    if args.model_name == 'MobileNetV2':
      model = MobileNetV2(args.image_length, True)
    if args.model_name == 'InceptionResNetV2':
      model = InceptionResNetV2(args.image_length, True)
    if args.model_name == 'ResNet152V2':
      model = ResNet152V2(args.image_length, True)
    if args.model_name == 'Model_1D2D':
      model = CNN_1D_2D_model(args.image_length, args.fft_length, True)

    name = 'model_' + args.model_name + '_' + args.based_image + '.h5'
    if args.load_weight:
      print(f'\nLoad weight file from {os.path.join(args.model_path, name)}\n')
      model.load_weights(os.path.join(args.model_path, name))
    # tf.keras.optimizers.RMSprop(1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc', sensitivity, specificity, average_score, harmonic_mean]) 
    model.summary()
    if args.train:
      if args.model_name == 'Model_1D2D':
        history = model.fit([images_org, ffts_org], labels_org,
                            epochs     = args.epochs,
                            batch_size = args.batch_size,)
      else:
        history = model.fit(image_train_data, train_label,
                            epochs     = args.epochs,
                            batch_size = args.batch_size,)
    if args.train:
      print(f'\nSave weight file to {os.path.join(args.model_path, name)}')
      model.save(os.path.join(args.model_path, name))
      
    ######################## TEST PHASE ##################################################################
    print('\n' + '-'*10 + 'Test phase' + '-'*10 + '\n') 
    # load neural network model
    if args.model_name == 'EfficientNetV2M':
      model = EfficientNetV2M(args.image_length, False)
    if args.model_name == 'MobileNetV2':
      model = MobileNetV2(args.image_length, False)
    if args.model_name == 'InceptionResNetV2':
      model = InceptionResNetV2(args.image_length, False)
    if args.model_name == 'ResNet152V2':
      model = ResNet152V2(args.image_length, False)
    if args.model_name == 'Model_1D2D':
      model = CNN_1D_2D_model(args.image_length, args.fft_length, False)
    
    if args.predict:
        # outputs validation by matrices: sensitivity, specificity, average_score, harmonic_mean
        model.load_weights(os.path.join(args.model_path, name))
        if args.model_name == 'Model_1D2D':
          pred_label = model.predict([image_test_data, test_fft])
        else:
          pred_label = model.predict(image_test_data)
        
        # Load matrices from predict data
        test_acc,  test_sensitivity,  test_specificity,  test_average_score, test_harmonic_mean  = matrices(test_label, pred_label)
        test_acc = round(test_acc*100, 2)
        test_sensitivity = round(test_sensitivity*100, 2)
        test_specificity = round(test_specificity*100, 2)
        test_average_score = round(test_average_score*100, 2)
        test_harmonic_mean = round(test_harmonic_mean*100, 2)
        print(f'\nAccuracy: {test_acc} \t SE: {test_sensitivity} \t SP: {test_specificity} \t AS: {test_average_score} \t HS: {test_harmonic_mean}\n')
        
        # display confution matrix
        test_label = np.argmax(test_label, axis=-1)
        pred_label = np.argmax(pred_label, axis=-1)
        cm = confusion_matrix(test_label, pred_label, labels=[0, 1, 2, 3])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'crackle', 'wheeze', 'both'])
        disp.plot()
        plt.title(args.model_name + ': ' + args.based_image)
        plt.savefig(args.model_path + '/images/' + 'model_' + args.model_name + '_' + args.based_image)
        plt.show()
        
if __name__ == "__main__":
    train(args)
