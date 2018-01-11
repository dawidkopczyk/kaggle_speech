from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop

from ctc_dataset import DatasetGenerator
from ctc_utils import ctc_dummy_loss, decoder_dummy_loss
from ctc_models import deep_ctc, deep_speech
from ctc_metrics import ler
from ctc_load import load_weights

# Data Loading
import os
import numpy as np

DIR = 'D:/Program Files/input/' # unzipped train and test data
OUT = './keras/' # just a random name
NET = 'ctc10'
SET2 = 'yes no up down left right on off stop go silence unknown'.split()
SET = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven sheila six stop three tree two up wow yes zero'.split()
BATCH = 64
EPOCHS = 50
SAMPLE = 2000
STRING_LEN = 6
            
dsGen = DatasetGenerator(label_set=SET, 
                         inputs_len=98, 
                         max_string_len=STRING_LEN,
                         sample=SAMPLE)  
dsGen.init_ggl_data(DIR)
dict_df = dsGen.get_df
df_train, df_val, df_test = dict_df['train'], dict_df['val'], dict_df['test']

num_classes = len(dsGen.get_alphabet) + 1
          
model = deep_ctc(features_shape=(177,98,1), num_classes=num_classes)
opt = RMSprop(lr=0.001)

model.compile(loss={'ctc': ctc_dummy_loss,
                    'decoder': decoder_dummy_loss},
              optimizer=opt, metrics={'decoder': ler},
              loss_weights=[1, 0])

callbacks = [EarlyStopping(monitor='val_decoder_ler',
                           patience=5,
                           verbose=1,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_decoder_ler',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=0.01,
                               mode='min'),
             ModelCheckpoint(monitor='val_decoder_ler',
                             filepath= OUT + NET + '.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')]

history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(df_train)/BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dsGen.generator(BATCH, mode='val'),
                              validation_steps=int(np.ceil(len(df_val)/BATCH)))

model_fname = OUT + NET + '.hdf5'
_model = load_weights(model, model_fname, mode='predict')

predictions = _model.predict_generator(dsGen.generator(BATCH, mode='test'), int(np.ceil(len(df_test)/BATCH)), verbose=1)

text_pred_raw = []
for i in range(predictions.shape[0]):
    translate = dsGen.labels_to_text(predictions[i])
    text_pred_raw.append(translate)

submission = dict()
for i in range(df_test.shape[0]):
    fname, label = os.path.basename(df_test.wav_file.values[i]), text_pred_raw[i]
    submission[fname] = label
    
with open('submission_raw_' + NET + '.csv', 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))