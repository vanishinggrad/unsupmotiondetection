#!/usr/bin/python3
# import os

random_seed = 1337
import random

random.seed(random_seed)
import numpy as np

np.random.seed(random_seed)
import tensorflow as tf

tf.random.set_seed(random_seed)

from tensorflow.keras import backend as K
from tensorflow.keras import metrics, regularizers, initializers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import wifi_data_config as conf
import argparse
import time

def get_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="if 1, run under training mode, if 0 run under test mode", type=int, default=1)
    args = parser.parse_args()
    return args



def get_classification_report(predict, truth, num_classes):
    results = np.zeros((num_classes, num_classes), np.float32)
    for k in range(predict.shape[0]):
        results[truth[k], predict[k]] += 1
    for k in range(num_classes):
        print('class {} has size {:.0f} static count {:.0f} motion count {:.0f}'.format(k, np.sum(results[k,:]), results[k,0], results[k,1]))
    results /= (np.sum(results, axis=1, keepdims=True) + 1e-6)
    for k in range(num_classes):
        acc = ' '.join(['{:.5f}']*num_classes)
        outstr = 'class {}: '.format(k)+acc.format(*results[k, :])
        print(outstr)

class NeuralNetworkModel:
    def __init__(self, data_shape, num_classes):
        self.model = None
        self.num_classes = num_classes
        self.data_shape = data_shape
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def cnn_model_phase(self, x):
        x = Conv2D(filters=12, kernel_size=(3, 3), strides=(1,1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(x)
        x = Conv2D(filters=12, kernel_size=(4, 4), strides=(1,1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(x)
        print("before flatten, shape of the phase data is: " + str(x.shape))
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(32,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        return x 

    def cnn_model_abs(self, x):
        x = Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(x)
        x = Conv2D(filters=12, kernel_size=(4, 4), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(x)
        print("before flatten, shape of the abs data is: " + str(x.shape))
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(32,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        return x 

    def cnn_model_iq(self):
        x_input = Input(shape=self.data_shape, name="main_input", dtype="float32")
        x = x_input
        x = Conv2D(filters=12, kernel_size=(5, 3), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(4, 1), strides=(4, 1))(x)
        x = Conv2D(filters=12, kernel_size=(4, 4), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(4, 1), strides=(4, 1))(x)
        print("before flatten, shape of the IQ data is: " + str(x.shape))
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(32,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='softmax', name="main_output")(x)
        self.model = Model(inputs=[x_input, ], outputs=x)



    def cnn_model_concatenate(self):
        x_input = Input(shape=self.data_shape, name="main_input", dtype="float32")
        x_abs = Lambda(lambda y: y[...,:, 0])(x_input)
        x_phase = Lambda(lambda y: y[...,:6, 1])(x_input)
        x = concatenate([x_abs, x_phase], axis=-1)
        x = Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(x)
        x = Conv2D(filters=12, kernel_size=(4, 4), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(x)
        print("before flatten, shape of the data is: " + str(x.shape))
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(32,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='softmax', name="main_output")(x)
        self.model = Model(inputs=[x_input, ], outputs=x)


    def cnn_model_abs_phase(self,):
        x_input = Input(shape=self.data_shape, name="main_input", dtype="float32")
        x_abs = Lambda(lambda y: y[...,:, 0], name='abs_input')(x_input)
        x_phase = Lambda(lambda y: y[...,:6, 1], name='phase_input')(x_input)
        x_abs = self.cnn_model_abs(x_abs)
        x_phase = self.cnn_model_phase(x_phase)
        x = concatenate([x_abs, x_phase])
        print("after concatenating, shape of the data is: " + str(x.shape))
        x = Dropout(0.5)(x)
        '''
        x = Dense(64,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = Dense(32,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        '''
        x = Dense(self.num_classes,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='softmax', name="main_output")(x)
        self.model = Model(inputs=[x_input, ], outputs=x)

    def dnn_model(self): 
        x_input = Input(shape=(128,), name="main_input", dtype="float32")
        x = Dense(64,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x_input)
        x = BatchNormalization()(x)
        x = Dense(self.num_classes,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='softmax', name="main_output")(x)
        self.model = Model(inputs=[x_input, ], outputs=x)


    def cnn_model(self,):
        x_input = Input(shape=self.data_shape, name="main_input", dtype="float32")
        print("first input shape to cnn " + str(self.data_shape))
        x = x_input
        x = Conv2D(filters=8, kernel_size=(1, 3), strides=(1, 3), padding='valid',
                   kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(1,2), strides=(1,2))(x)
        x = Conv2D(filters=9, kernel_size=(3, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(x)
        print("before flatten, shape of the data is: " + str(x.shape))
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(64,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(self.num_classes,
                  kernel_regularizer=regularizers.l2(0.02),
                  kernel_initializer=initializers.glorot_uniform(),
                  activation='softmax', name="main_output")(x)
        self.model = Model(inputs=[x_input, ], outputs=x)


    def fit_data(self,epochs):
        train_num = {}
        for m in range(self.num_classes):
            train_num[m] = 0
        for m in range(self.y_train.shape[0]):
            train_num[self.y_train[m,0]] += 1
        print("training data {}".format(train_num))
        class_weight = {}
        for m in range(self.num_classes):
            class_weight[m] = (self.y_train.shape[0]-train_num[m])/float(self.y_train.shape[0])

        print("class weight {}".format(class_weight))
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)
        Op = Adam(lr=0.001, decay=0.005, beta_1=0.9, beta_2=0.999)
        #Op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model.summary()
        start_t = time.time()
        self.model.compile(optimizer=Op, loss=['categorical_crossentropy', ],
                            metrics=[metrics.categorical_accuracy])
        self.model.fit(x=self.x_train, y=self.y_train,
                       epochs=epochs, verbose=1, batch_size=256, shuffle=True,
                       validation_data=(self.x_test, self.y_test))
        end_t = time.time()
        print('training time: {:.6f}'.format(end_t-start_t))

    def save_model(self, model_name):
        self.model.save(model_name)
        print("trained mode was saved as {} successfully\n".format(model_name))

    def load_model(self, model_name):
        self.model = load_model(model_name)
        # self.model.summary()
        '''
        for layer in self.model.layers[:-6]:
            layer.trainable = False
        '''
        print("model {} was loaded successfully\n".format(model_name))

    def predict(self, data, batch_size=128):
        p = self.model.predict(data, batch_size=batch_size)
        return p[:,-1]
        # return np.argmax(p, axis=-1)

    def end(self):
        K.clear_session()

    def add_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def transform(self, b):
        ampl = b[..., 0]
        phase = b[...,:6, 1]
        b[...,0] = (ampl - np.amin(ampl, axis=(1,2), keepdims=True))/(np.amax(ampl, axis=(1,2), keepdims=True)-np.amin(ampl, axis=(1,2), keepdims=True))
        b[...,:6, 1] = (phase - np.amin(phase, axis=(1,2), keepdims=True))/(np.amax(phase, axis=(1,2), keepdims=True)-np.amin(phase, axis=(1,2), keepdims=True))
        return b

    def draw_result(self,x, y):
        l = np.squeeze(y)  
        static_label = np.where(l==0)[0]
        static_index = np.random.permutation(len(static_label))
        motion_label = np.where(l==1)[0]
        motion_index = np.random.permutation(len(motion_label))
        cc = 5
        for m in range(cc):
            for k in range(1):
                plt.figure()
                plt.plot(x[static_label[static_index[m]], :])
                #plt.matshow(x[static_label[static_index[m]], :, :, k, 0])
                plt.title('static '+str(k))
        for m in range(cc):
            for k in range(1):
                #plt.matshow(x[motion_label[motion_index[m]], :, :, k, 0])
                plt.figure()
                plt.plot(x[motion_label[motion_index[m]], :])
                plt.title('motion '+str(k))
        plt.show()

    def shrink(self, d, l):
        s_index = 0
        for m in range(d.shape[0]):
            if l[m]:
                s_index = m
                break
        static_index = list(range(s_index))
        presence_num = d.shape[0] - s_index
        static_num = s_index
        remove_num = presence_num - static_num
        m = np.random.permutation(presence_num)+s_index
        presence_index = m[:static_num]
        final_index = static_index + list(presence_index)
        return d[final_index, :], l[final_index, :]

    def get_data_from_file(self, file_prefix, data_type, training_mode):
        if training_mode:
            filename = file_prefix + 'x_train.dat'
            temp_image = np.fromfile(filename, dtype=data_type)
            self.x_train = np.reshape(temp_image, (-1,) + self.data_shape)
            filename = file_prefix + 'y_train.dat'
            temp_label = np.fromfile(filename, dtype=np.int8)
            self.y_train = np.reshape(temp_label, (-1, 1))
            '''
            m = np.random.permutation(self.x_train.shape[0])
            self.x_train = self.x_train[m[:40000],...]
            self.y_train = self.y_train[m[:40000]]
            filename = './data/nobhill_data/x_train.dat'
            temp_image = np.fromfile(filename, dtype=data_type)
            self.x_train = np.concatenate([self.x_train, np.reshape(temp_image, (-1,) + self.data_shape)],axis=0)
            filename = './data/nobhill_data/y_train.dat'
            temp_label = np.fromfile(filename, dtype=np.int8)
            self.y_train = np.concatenate([self.y_train, np.reshape(temp_label, (-1, 1))], axis=0)
            '''
        filename = file_prefix + 'x_test.dat'
        temp_image = np.fromfile(filename, dtype=data_type)
        self.x_test = np.reshape(temp_image, (-1,) + self.data_shape)
        filename = file_prefix + 'y_test.dat'
        temp_label = np.fromfile(filename, dtype=np.int8)
        self.y_test = np.reshape(temp_label, (-1, 1))
        '''
        filename = './2nd_floor_data/x_test.dat'
        temp_image = np.fromfile(filename, dtype=data_type)
        self.x_test = np.concatenate([self.x_test, np.reshape(temp_image, (-1,) + self.data_shape)],axis=0)
        filename = './2nd_floor_data/y_test.dat'
        temp_label = np.fromfile(filename, dtype=np.int8)
        self.y_test = np.concatenate([self.y_test, np.reshape(temp_label, (-1, 1))], axis=0)
        '''

    def get_test_result(self):
        if self.y_test.shape[-1] > 1:
            self.y_test = np.argmax(self.y_test, axis=-1)
        start_t = time.time()
        p = self.predict(self.x_test, batch_size=1)
        end_t = time.time()
        print('test time: {:.6f}'.format(end_t-start_t))
        p = p.astype('int8')
        get_classification_report(p, self.y_test, self.num_classes)
        return p
    
    def get_training_result(self):
        if self.y_train.shape[-1] > 1:
            self.y_train = np.argmax(self.y_train, axis=-1) 
        p = self.predict(self.x_train, batch_size=128)
        p = p.astype('int8')
        get_classification_report(p, self.y_train, self.num_classes)
        return p
 

    def get_no_label_result(self, dd, batch_size=128):
        p = self.predict(dd, batch_size=batch_size)
        #p = p.astype('int8')
        p = p.astype('float32')
        return p

    def get_model(self):
        return self.model

    def save_result(self, p, filename):
        p.tofile(filename)
        print("test result was saved to "+filename+"\n")

def main():
    args = get_input_arguments()
    training_mode = args.mode
    nn_model = NeuralNetworkModel(conf.data_shape_to_nn, conf.total_classes)
    nn_model.get_data_from_file(conf.file_prefix, np.float32, training_mode)
    if training_mode:
        if conf.use_existing_model:
            nn_model.load_model(conf.model_name)
        else:
            nn_model.cnn_model_abs_phase()
        nn_model.fit_data(conf.epochs)
        print('training result: ')
        train_result = nn_model.get_training_result()
        print('get result: ')
        test_result = nn_model.get_test_result()
        nn_model.save_model(conf.model_name)

    else:
        nn_model.load_model(conf.model_name)
        print('testing result: ')
        result = nn_model.get_test_result()  
        nn_model.save_result(result, conf.file_prefix+conf.test_result_filename)
    nn_model.end()

if __name__ == "__main__":
    main()
