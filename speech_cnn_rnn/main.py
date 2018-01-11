DATADIR = 'D:/Program Files/input/' # unzipped train and test data
OUTDIR = './keras/' # just a random name
NETNAME = 'lenet1'

# Data Loading
import os
import re
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import stft
from scipy.fftpack import fft
from scipy import signal

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models import LeNet, MyNet, CNN, CNN_RNN, CNN_RNN2, CNN_RNN3

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    all_files = [x.split(sep='\\')[1] + '/' + x.split(sep='\\')[2] for x in all_files]
    
    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            entry = os.path.join(data_dir,'train/audio/',entry)
            
            sample = (label, label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    columns_list = ['label', 'label_id', 'user_id', 'wav_file']
        
    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(val, columns = columns_list)
    
    return train_df, valid_df

train_df, valid_df = load_data(DATADIR)

train_df.head()
train_df.label.value_counts()

silence_files = train_df[train_df.label == 'silence']
train_df      = train_df[train_df.label != 'silence']
silence_df = silence_files.iloc[np.random.randint(0, len(silence_files), size=2000)]
train_df = train_df.append(silence_df, ignore_index=True)


def read_wav_file(fname, noise_flag=False):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])

def process_wav_file(fname, noise_flag=False):
    wav = read_wav_file(fname)
    
    L = 16000  # 1 sec
    
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
#    freqs, times, spec = stft(wav, L, nperseg = 192, noverlap = 160, padded = False, boundary = None)
    spec = spec[freqs <= 5500,:]
    freqs = freqs[freqs <= 5500]
#    phase = np.angle(spec) / np.pi
    amp = np.log(np.abs(spec)+1e-10)

#    fig = plt.figure(figsize=(14, 8))
#    ax1 = fig.add_subplot(211)
#    ax1.set_title('Raw wave')
#    ax1.set_ylabel('Amplitude')
#    ax1.plot(np.linspace(0, L/len(wav), L), wav)
#    
#    ax2 = fig.add_subplot(212)
#    ax2.imshow(amp, aspect='auto', origin='lower', 
#               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
#    ax2.set_yticks(freqs[::16])
#    ax2.set_xticks(times[::16])
#    ax2.set_title('Spectrogram')
#    ax2.set_ylabel('Freqs in Hz')
#    ax2.set_xlabel('Seconds')
  
#    return np.stack([phase, amp], axis = 2)
    return np.expand_dims(amp, axis=2)

#xx = process_wav_file(train_df.wav_file.values[0])
#xx.shape

import random
from keras.utils import to_categorical

def train_generator(train_batch_size):
    while True:
#        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        this_train = train_df
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i], False))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch
            
def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i]))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch

test_paths = glob(os.path.join(DATADIR, 'test/audio/*wav'))

def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x))
            x_batch = np.array(x_batch)
            yield x_batch

from keras.optimizers import RMSprop
from keras.layers import LeakyReLU

params = {'activation': ['ELU'],
        'name': ['CNN_RNN4_sil_trainall_nophase']}

for param_idx in range(len(params['name'])):          
    model = CNN_RNN4.build(features_shape=(177,98,1), num_classes=len(POSSIBLE_LABELS), optimizer=RMSprop(lr=0.001), activation='relu')
    
    callbacks = [EarlyStopping(monitor='val_acc',
                               patience=5,
                               verbose=1,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_acc',
                                   factor=0.1,
                                   patience=3,
                                   verbose=1,
                                   epsilon=0.01,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_acc',
                                 filepath= OUTDIR + params['name'][param_idx] + '.hdf5',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')]
    
    history = model.fit_generator(generator=train_generator(64),
                                  steps_per_epoch=int(np.ceil(train_df.shape[0])/64),
                                  epochs=50,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=valid_generator(64),
                                  validation_steps=int(np.ceil(valid_df.shape[0]/64)))
    
    model.load_weights(OUTDIR + params['name'][param_idx] + '.hdf5')
         
    predictions = model.predict_generator(test_generator(64), int(np.ceil(len(test_paths)/64)), verbose=1)
    np.save('pred_' + params['name'][param_idx], predictions)
    classes = np.argmax(predictions, axis=1)
    
    # last batch will contain padding, so remove duplicates
    submission = dict()
    for i in range(len(test_paths)):
        fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
        submission[fname] = label
        
    with open('submission_' + params['name'][param_idx] + '.csv', 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write('{},{}\n'.format(fname, label))