import os
import re
import numpy as np
import pandas as pd
import random
import librosa
import webrtcvad
import struct
import tensorflow as tf
    
from glob import glob

from scipy.sparse import coo_matrix
from scipy.io import wavfile
from scipy.signal import stft

class DatasetGenerator():
    def __init__(self, label_set, inputs_len, max_string_len=5, 
                 sample_rate=16000, sample=2000,
                 pattern = "(.+\/)?(\w+)\/([^_]+)_.+wav"):
        
        self.label_set = label_set
        self.inputs_len = inputs_len
        self.max_string_len = max_string_len
        self.sample_rate = sample_rate
        self.sample = sample
        self.pattern = re.compile(pattern)
        
        self.silence_dir = '_background_noise_'
        self.silence_label = 'silence'
        self.silence_char = ' '
#        self.unknown_label = 'unknown'
#        self.unknown_char = '$'
        self.alphabet = self._create_alphabet()
        
    def _create_alphabet(self):
        alphabet = set(list('abcdefghijklmnopqrstuvwxyz'))
        existing = set()
        for i, label in enumerate(self.label_set):
            if label == self.silence_label:
                pass
            else:
                for char in label:
                    existing.add(char)
        diff = alphabet - existing
        for char in diff:
            alphabet.remove(char)
        return ''.join(sorted(alphabet))+self.silence_char
    
    # Covert string to numerical classes              
    def text_to_labels(self, text):
        ret = []
        for char in text:
            ret.append(self.alphabet.find(char))
        return ret
    
    # Reverse translation of numerical classes back to characters
    def labels_to_text(self, labels):
        ret = []
        for c in labels:
            if c == len(self.alphabet):  # CTC Blank
                ret.append('')
            else:
                ret.append(self.alphabet[c])
        return ''.join(ret)                
                        
        
    def _load_ggl_data(self, wav_dir, wav_paths, valset):
    
        train, val = [], []
        for e in wav_paths:
            r = re.match(self.pattern, e)
            if r:
                label, uid = r.group(2), r.group(3)
                if label == self.silence_dir:
                    label = self.silence_label
                    label_id = self.text_to_labels(self.silence_char)
#                elif label == self.unknown_label or label not in self.label_set:
#                    label = self.unknown_label 
#                    label_id = self.text_to_labels(self.unknown_char)
                else:
                    label_id = self.text_to_labels(label)
            
                fle = os.path.join(wav_dir, 'train/audio/', e)
                
                sample = (label, label_id, uid, fle)
                if uid in valset:
                    val.append(sample)
                else:
                    train.append(sample)
    
        columns_list = ['label', 'label_id', 'user_id', 'wav_file']
            
        df_train = pd.DataFrame(train, columns = columns_list)
        df_val = pd.DataFrame(val, columns = columns_list)
        
        return df_train, df_val
    
    def init_ggl_data(self, path, add_silence=True):
        all_files = glob(os.path.join(path, 'train/audio/*/*wav'))
        all_files = [x.split(sep='\\')[1] + '/' + x.split(sep='\\')[2] for x in all_files]
        
        with open(os.path.join(path, 'train/validation_list.txt'), 'r') as fin:
            val_files = fin.readlines()
        valset = set()
        for f in val_files:
            r = re.match(self.pattern, f)
            if r:
                valset.add(r.group(3))
        
        df_train, df_val = self._load_ggl_data(path, all_files, valset)

        df_silence = df_train[df_train.label == self.silence_label]
        df_train   = df_train[df_train.label != self.silence_label]
        if add_silence:
            df_silence_add = df_silence.iloc[np.random.randint(0, df_silence.shape[0], size=2000)]
            df_train = df_train.append(df_silence_add, ignore_index=True)

        self.silence_data = np.concatenate([self.read_wav_file(x) for x in df_silence.wav_file.values])
#        self.df_train = pd.concat([df_train]*1, ignore_index=True)
        self.df_train = df_train
        self.df_val = df_val
        self.df_silence = df_silence
        
        test = glob(os.path.join(path, 'test/audio/*wav'))
        self.df_test = pd.DataFrame(test, columns = ['wav_file'])
    
    # stretching the sound
    def stretch(self, wav, rate=1):
        wav = librosa.effects.time_stretch(wav, rate)
        if len(wav)>self.sample_rate:
            wav = wav[:self.sample_rate]
    
        return wav

    def read_wav_file(self, x, mode='train'):
        _, wav = wavfile.read(x) 
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        
#        if not len(wav) > self.sample_rate and mode == 'train':
#            
#            noise_flag = np.random.random(1) > 0.5
#            roll_flag = np.random.random(1) > 0.5
#            stretch_flag = np.random.random(1) > 0.5
#            
#            if noise_flag:
#                wn = np.random.randn(len(wav))
#                wav = wav + 0.005*wn
#                
#            if roll_flag:
#                wav = np.roll(wav, int(np.random.choice([-1,1]) * self.sample_rate / 10))
#                
#            if stretch_flag:
#                wav = self.stretch(wav, np.random.choice([0.8,1.2]))
            
        return wav
    
    def process_wav_file(self, x, mode='train', threshold_freq=5500, eps=1e-10):
        wav = self.read_wav_file(x, mode)
        
        L = self.sample_rate
        
        if len(wav) > L:
            i = np.random.randint(0, len(wav) - L)
            wav = wav[i:(i+L)]  
        elif len(wav) < L:
            rem_len = L - len(wav)
            silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
            j = np.random.randint(0, rem_len)
            silence_part_left  = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])
    
        freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq,:]
            freqs = freqs[freqs <= threshold_freq]
#        phase = np.angle(spec) / np.pi
        amp = np.log(np.abs(spec)+eps)
    
        return np.expand_dims(amp, axis=2)
#        return np.stack([phase, amp], axis = 2)  

    def generator(self, batch_size, mode='train'):
        while True:
            
            if mode == 'train':
#                df = self.df_train.groupby('label').apply(lambda x: x.sample(n = 2000))
                df = self.df_train 
                ids = random.sample(range(df.shape[0]), df.shape[0])
            elif mode == 'val':
                df = self.df_val
                ids = list(range(df.shape[0]))
            elif mode == 'test':
                df = self.df_test
                ids = list(range(df.shape[0]))
            else:
                raise ValueError('The mode should be either train, val or test.')
                
            for start in range(0, len(ids), batch_size):
                x_batch = []
                if mode != 'test': 
                    y_batch = []
                inputs_len = []
                end = min(start + batch_size, len(ids))
                i_batch = ids[start:end]
                for i in i_batch:
                    x_batch.append(self.process_wav_file(df.wav_file.values[i], mode))
                    if mode != 'test':
                        y_batch.append(df.label_id.values[i])
                    inputs_len.append(self.inputs_len)
                x_batch = np.array(x_batch)
                if mode != 'test':
                    rows, cols, data = [], [], []
                    for row, lab in enumerate(y_batch):
                        cols.extend(range(len(lab)))
                        rows.extend(len(lab) * [row])
                        data.extend(lab)
                    y_batch = coo_matrix((data, (rows, cols)), shape=(x_batch.shape[0], self.max_string_len), dtype='int32')     
                inputs_len = np.expand_dims(np.array(inputs_len), 1)
                
                if mode != 'test':
                    inputs = {'inputs': x_batch,
                              'labels': y_batch,
                              'inputs_length': inputs_len}
                    outputs = {'ctc': np.zeros([x_batch.shape[0]]),
                               'decoder': y_batch}
                    yield (inputs, outputs)
                else:
                    inputs = {'inputs': x_batch,
                              'inputs_length': inputs_len}
                    yield inputs
        
    @property
    def get_df(self):
        return {'train': self.df_train, 
                'val': self.df_val, 
                'test': self.df_test, 
                'silence': self.df_silence}
 
    @property
    def get_alphabet(self):
        return self.alphabet
                                       