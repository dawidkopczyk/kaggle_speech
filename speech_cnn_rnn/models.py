from keras.models import Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, 
                          AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPool2D,
                          concatenate, Dense, Dropout, Flatten, Merge, Permute, GRU, TimeDistributed, Reshape, Bidirectional)
from keras.models import Model
from keras.optimizers import SGD

class LeNet:
    @staticmethod
    def build(features_shape, num_classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, 5, 5, padding="same",
            input_shape=features_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, 5, 5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['acc'])
            
        return model
    
class MyNet():
    @staticmethod
    def build(features_shape, num_classes, optimizer='rmsprop', weightsPath=None):
        x_in = Input(shape = features_shape)
        x = BatchNormalization()(x_in)
        for i in range(4):
            x = Conv2D(16*(2 ** i), (3,3))(x)
            x = Activation('elu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2,2))(x)
        x = Conv2D(128, (1,1))(x)
        x_branch_1 = GlobalAveragePooling2D()(x)
        x_branch_2 = GlobalMaxPool2D()(x)
        x = concatenate([x_branch_1, x_branch_2])
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation = 'sigmoid')(x)
        model = Model(inputs = x_in, outputs = x)
        
        model.summary()
       
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        
        return model

class VGG16():
    
    @staticmethod
    def build(features_shape, num_classes, weightsPath=None):
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=features_shape))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    
        # Block 2
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
        # Block 3
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
#        # Block 4
#        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1'))
#        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2'))
#        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3'))
#        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
#    
#        # Block 5
#        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1'))
#        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2'))
#        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3'))
#        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(256, activation='relu', name='fc1'))
        model.add(Dense(256, activation='relu', name='fc2'))
        model.add(Dense(num_classes, activation='softmax', name='predictions'))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
            
        return model
    
class CNN():
    
    @staticmethod
    def build(features_shape, num_classes, optimizer='rmsprop', neurons=512, activation='relu', weightsPath=None):
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(16, (7, 7), activation='relu', padding='same', strides=2, name='block1_conv1', input_shape=features_shape))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block1_pool'))
        model.add(BatchNormalization(name='block1_norm'))
        
        # Block 2
        model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=1, name='block2_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block2_pool'))
        model.add(BatchNormalization(name='block2_norm'))

        # Block 3
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='block3_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block3_pool'))
        model.add(BatchNormalization(name='block3_norm'))

        # Block 4
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=1, name='block4_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block4_pool'))
        model.add(BatchNormalization(name='block4_norm'))
        
        # Block 5
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=1, name='block5_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block5_pool'))
        model.add(BatchNormalization(name='block5_norm'))        

        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(neurons, activation=activation, name='fc1'))
        model.add(BatchNormalization(name='fc1_norm'))
        model.add(Dropout(0.5, name='fc1_drop'))
        model.add(Dense(num_classes, activation='softmax', name='pred'))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            
        return model
    
class CNN_RNN():
    
    @staticmethod
    def build(features_shape, num_classes, optimizer='rmsprop', neurons=512, activation='relu', weightsPath=None):
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(16, (7, 7), activation='relu', padding='same', strides=2, name='block1_conv1', input_shape=features_shape))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block1_pool'))
        model.add(BatchNormalization(name='block1_norm'))
        
        # Block 2
        model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=1, name='block2_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block2_pool'))
        model.add(BatchNormalization(name='block2_norm'))

        # Block 3
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, name='block3_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block3_pool'))
        model.add(BatchNormalization(name='block3_norm'))       

        # RNN block
        model.add(Permute((3, 2, 1), name='rnn_permute'))
        model.add(Reshape((32*7, 12), name='rnn_reshape'))
        model.add(GRU(units=128, name='rnn_gru'))
        model.add(BatchNormalization(name='rnn_norm'))
        
        # Classification block
#        model.add(Flatten(name='flatten'))
#        model.add(Dense(neurons, activation=activation, name='fc1'))
#        model.add(BatchNormalization(name='fc1_norm'))
        model.add(Dropout(0.5, name='fc1_drop'))
        model.add(Dense(num_classes, activation='softmax', name='pred'))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            
        return model
    
class CNN_RNN2():
    
    @staticmethod
    def build(features_shape, num_classes, optimizer='rmsprop', neurons=512, activation='relu', weightsPath=None):
        model = Sequential()
        
                # Block 1
        model.add(Conv2D(16, (7, 7), activation='relu', padding='same', strides=2, name='block1_conv1', input_shape=features_shape))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block1_pool'))
        model.add(BatchNormalization(name='block1_norm'))
        
        # Block 2
        model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=1, name='block2_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block2_pool'))
        model.add(BatchNormalization(name='block2_norm'))

        # Block 3
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, name='block3_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block3_pool'))
        model.add(BatchNormalization(name='block3_norm'))

        # Block 4
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=1, name='block4_conv1'))
        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block4_pool'))
        model.add(BatchNormalization(name='block4_norm'))
        
        # Block 5
#        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=1, name='block5_conv1'))
#        model.add(MaxPooling2D((3, 3), strides=2, padding='same', name='block5_pool'))
#        model.add(BatchNormalization(name='block5_norm'))   

        # RNN block
        model.add(Permute((2, 1, 3), name='rnn_permute'))
        model.add(Reshape((4, 6*128), name='rnn_reshape'))
        model.add(Bidirectional(GRU(units=512), name='rnn_bigru'))
        model.add(BatchNormalization(name='rnn_norm'))
        model.add(Dropout(0.5, name='rnn_drop'))
        
        # Classification block 
        model.add(Dense(num_classes, activation='softmax', name='pred'))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            
        return model
    
class CNN_RNN3():
    
    @staticmethod
    def build(features_shape, num_classes, optimizer='rmsprop', neurons=512, activation='relu', weightsPath=None):
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(16, (7, 7), activation=activation, padding='same', strides=1, name='block1_conv1', input_shape=features_shape))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block1_pool'))
        model.add(BatchNormalization(name='block1_norm'))
        
        # Block 2
        model.add(Conv2D(32, (5, 5), activation=activation, padding='same', strides=1, name='block2_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block2_pool'))
        model.add(BatchNormalization(name='block2_norm'))

        # Block 3
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', strides=1, name='block3_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block3_pool'))
        model.add(BatchNormalization(name='block3_norm'))

        # Block 4
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', strides=1, name='block4_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block4_pool'))
        model.add(BatchNormalization(name='block4_norm'))
        
        # Block 5
#        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, name='block5_conv1'))
#        model.add(MaxPooling2D((3, 3), strides=1, padding='same', name='block5_pool'))
#        model.add(BatchNormalization(name='block5_norm'))   

        # RNN block
        model.add(Permute((2, 1, 3), name='rnn_permute'))
        model.add(Reshape((98, 12*32), name='rnn_reshape'))
        model.add(Bidirectional(GRU(units=128), name='rnn_bigru'))
        model.add(BatchNormalization(name='rnn_norm'))
        model.add(Dropout(0.5, name='rnn_drop'))
        
        # Classification block 
        model.add(Dense(num_classes, activation='softmax', name='pred'))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            
        return model
    
class CNN_RNN4():
    
    @staticmethod
    def build(features_shape, num_classes, optimizer='rmsprop', neurons=512, activation='relu', weightsPath=None):
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', strides=1, name='block1_conv1', input_shape=features_shape))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block1_pool'))
        model.add(BatchNormalization(name='block1_norm'))
        
        # Block 2
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', strides=1, name='block2_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block2_pool'))
        model.add(BatchNormalization(name='block2_norm'))

        # Block 3
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', strides=1, name='block3_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block3_pool'))
        model.add(BatchNormalization(name='block3_norm'))

        # Block 4
        model.add(Conv2D(32, (3, 3), activation=activation, padding='same', strides=1, name='block4_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block4_pool'))
        model.add(BatchNormalization(name='block4_norm'))
        
        # Block 5
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, name='block5_conv1'))
        model.add(MaxPooling2D((3, 3), strides=(2,1), padding='same', name='block5_pool'))
        model.add(BatchNormalization(name='block5_norm'))   

        # RNN block
        model.add(Permute((2, 1, 3), name='rnn_permute'))
        model.add(Reshape((98, 6*32), name='rnn_reshape'))
        model.add(Bidirectional(GRU(units=512), name='rnn_bigru'))
        model.add(BatchNormalization(name='rnn_norm'))
#        model.add(Dropout(0.5, name='rnn_drop'))
        
        # Classification block 
        model.add(Dense(num_classes, activation='softmax', name='pred'))
        
        model.summary()
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        else:   
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            
        return model