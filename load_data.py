import os
from utils.tools import to_onehot, load_df, create_spectrograms_raw, \
                        get_annotations, get_sound_samples, save_df, sensitivity, \
                        specificity, average_score, harmonic_mean, \
                        matrices, create_stft, mix_up, convert_fft, power_spectrum, arrange_data
import progressbar


def load_mel(args, train_data, test_data):
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
  return image_train_data, image_test_data

def load_stft(args, train_data, test_data):
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
  return image_train_data, image_test_data
