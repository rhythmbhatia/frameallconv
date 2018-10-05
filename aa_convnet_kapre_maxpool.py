from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras
import kapre
import librosa
from librosa import display
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)

input_shape = (1, 8820)
sr = 44100   
                  
model = Sequential()
#model.add(Melspectrogram(n_dft=1024, n_hop=882, input_shape=input_shape,
#                         padding='same', sr=sr, n_mels=40,
#                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
#                         return_decibel_melgram=True, trainable_fb=True,
#                         trainable_kernel=True, image_data_format='channels_last',
#                         name='trainable_stft'))  
model.add(Melspectrogram(sr=sr, n_mels=40, 
          n_dft=1024, n_hop=882*2, input_shape=input_shape, 
          return_decibel_melgram=True,trainable_fb=False,
          trainable_kernel=False, name='melgram'))                       
#model.add(AdditiveNoise(power=0.2))
model.add(Normalization2D(str_axis='freq'))
#conv1
#model.add(ZeroPadding2D((2,2))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu',padding='same', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool1
model.add(Conv2D(16, (5, 1), strides=(5, 1), padding='same', activation='relu'))
model.add(Dropout(0.50))

#conv2
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool2
model.add(Conv2D(16, (2, 1), strides=(2, 1), activation='relu',padding='same'))
model.add(Dropout(0.50))

#conv3
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu',padding='same', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool3
model.add(Conv2D(16, (2, 1), strides=(2, 1), activation='relu',padding='same'))
model.add(Dropout(0.50))

#conv4
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu',padding='same', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool4
model.add(Conv2D(16, (2, 1), strides=(2, 1), activation='relu',padding='same'))
model.add(Dropout(0.50))

#stacking(reshaping)
model.add(Reshape((5, 16)))

#learnedpool5
model.add(Conv1D(16, (5), strides=(1), activation='relu'))
model.add(Dropout(0.50))

#fully connected layers using conv
model.add(Reshape((1, 1,16)))
model.add(Conv2D(196,(1,1),activation = 'sigmoid'))
model.add(Dropout(0.50))
model.add(Conv2D(2,(1,1),activation = 'softmax'))
model.add(Reshape((2,)))
model.summary()

#train_data
classes = 2
feature = np.load('/home/birds/arjun/rhythm_music/UntitledFolder/new_features/featurenew.npy')
print(feature.shape)
#eature='/home/birds/arjun/rhythm_music/UntitledFolder/features/MIR/all_data'
label = np.load('/home/birds/arjun/rhythm_music/UntitledFolder/new_features/labelnew.npy')
label = to_categorical(label, 2)
adm = keras.optimizers.Adam(lr=0.01, decay=1e-6)
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.40, shuffle=True)

# compile model
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['categorical_accuracy'])
#fit the model
#filepath="kapre_music_Seg_batch96_40ms.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=2, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]
hist = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=200, batch_size=32,verbose=2)

#save_model
#model.save('/home/birds/arjun/rhythm_music/UntitledFolder/kapre_convnet.h5') 

