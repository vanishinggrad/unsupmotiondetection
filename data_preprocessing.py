#!/usr/bin/python3

import argparse

import numpy as np

import wifi_data_config as conf
from global_sp_func import sp_func, reshape_func, shape_conversion


def get_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="if 1, run under training mode, if 0 run under test mode", type=int, default=1)
    args = parser.parse_args()
    return args



def append_array(array_a, array_b, axis=0):
    if array_a.size == 0:
        array_a = array_a.astype(array_b.dtype)
        array_a = array_a.reshape((0,) + array_b.shape[1:])
    array_a = np.concatenate([array_a, array_b], axis)
    return array_a


class DataPreprocess:
    def __init__(self, n_timestamps, D, step_size, ntx_max,
                 ntx, nrx_max, nrx, nsubcarrier_max, nsubcarrier, output_shape, file_prefix, label):
        self.file_prefix = file_prefix
        self.data_shape = (n_timestamps, nrx_max, ntx_max, nsubcarrier_max)
        self.step_size = step_size
        self.n_timestamps = n_timestamps
        self.ntx = ntx
        self.nrx = nrx
        self.nsubcarrier = nsubcarrier
        self.subcarrier_spacing = int(nsubcarrier_max/nsubcarrier)
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_test = np.array([])
        self.no_label_test = None
        self.x_evaluate = {}
        self.classes_num = {}
        self.label = label
        self.output_shape = output_shape


    def add_image_no_label(self, test_data_class):
        for key in test_data_class:
            for idx in test_data_class[key]:
                test_data_class[key][idx] = reshape_func(test_data_class[key][idx], self.subcarrier_spacing)
        self.no_label_test = test_data_class

    def load_image(self, training_mode, from_file, train_data_class={}, test_data_class={}):
        x_train = np.array([])
        x_test = np.array([])
        y_train = np.array([])
        y_test = np.array([])
        self.classes_num = {}
        for k,o in self.label:
            if o not in self.classes_num.keys():
                self.classes_num[o] = {'train_num': 0, 'test_num': 0}
            if training_mode:
                if from_file:
                    filename = self.file_prefix+'training_'+str(o)+'.dat'
                    print('train filename '+filename)
                    temp_image = np.fromfile(filename, dtype=np.complex64)
                    temp_image = np.reshape(temp_image, (-1,) + self.data_shape)
                else:
                    temp_image = train_data_class[o]
                if temp_image.shape[0] != 0:
                    temp_image = reshape_func(temp_image, self.subcarrier_spacing)
                    temp_label = np.full((temp_image.shape[0], 1), o, np.int8)
                    self.classes_num[o]['train_num'] += temp_label.shape[0]
                    x_train = append_array(x_train, temp_image)
                    y_train = append_array(y_train, temp_label)
                if from_file:
                    test_filename = self.file_prefix+'training_test_'+str(o)+'.dat'
            else:
                if from_file:
                    test_filename = self.file_prefix+'test_'+str(o)+'.dat'
            if from_file:
                print('test filename '+test_filename)
                temp_image = np.fromfile(test_filename, dtype=np.complex64)
                temp_image = np.reshape(temp_image, (-1,) + self.data_shape)
            else:
                temp_image = test_data_class[o]
            if temp_image.shape[0] == 0:
                continue
            temp_image = reshape_func(temp_image, self.subcarrier_spacing)
            temp_label = np.full((temp_image.shape[0], 1), o, np.int8)
            self.classes_num[o]['test_num'] += temp_label.shape[0]
            x_test = append_array(x_test, temp_image)
            y_test = append_array(y_test, temp_label)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        print(self.classes_num)
        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError('x_train and y_train size mismatch')
        if self.x_test.shape[0] != self.y_test.shape[0]:
            raise ValueError('x_test and y_test size mismatch')



    def reshape_image(self):
        if self.x_train.shape[0]>0:
            self.x_train = reshape_func(self.x_train, self.subcarrier_spacing)
        if self.x_test.shape[0]>0:
            self.x_test = reshape_func(self.x_test, self.subcarrier_spacing)
        if self.no_label_test is not None:
            for key in self.no_label_test:
                for idx in self.no_label_test[key]:
                    self.no_label_test[key][idx] = reshape_func(self.no_label_test[key][idx], self.subcarrier_spacing)

    def signal_processing(self, do_fft, fft_shape):
        if self.x_train.shape[0]>0:
            self.x_train = sp_func(self.x_train, do_fft, fft_shape)
        if self.x_test.shape[0]>0:
            self.x_test = sp_func(self.x_test, do_fft, fft_shape)
        if self.no_label_test is not None:
            for key in self.no_label_test:
                for idx in self.no_label_test[key]:
                    self.no_label_test[key][idx] = sp_func(self.no_label_test[key][idx], do_fft, fft_shape)

    def norm_function(self, d):
        d = d/d[:, :1, ...]
        return d

    def normalize_image(self):
        #print('normalization is called!')
        if self.x_train.shape[0]>0:
            self.x_train = self.norm_function(self.x_train)
        if self.x_test.shape[0]>0:
            self.x_test = self.norm_function(self.x_test)
        if self.no_label_test is not None:
            for key in self.no_label_test:
                for idx in self.no_label_test[key]:
                    self.no_label_test[key][idx] = self.norm_function(self.no_label_test[key][idx])

    def prepare_shape(self):
        if self.x_train.shape[0]:
            self.x_train = shape_conversion(self.x_train, self.output_shape[0])
            print('final training data shape {}'.format(self.x_train.shape))
        if self.x_test.shape[0]:
            self.x_test = shape_conversion(self.x_test, self.output_shape[0])
            print('final test data shape {}'.format(self.x_test.shape))
        if self.no_label_test is not None:
            for key in self.no_label_test:
                for idx in self.no_label_test[key]:
                    self.no_label_test[key][idx] = shape_conversion(self.no_label_test[key][idx], self.output_shape[0])

    def save2file(self):
        if self.x_train.shape[0]>0:
            self.x_train.tofile(self.file_prefix+"x_train.dat")
            self.y_train.tofile(self.file_prefix+"y_train.dat")
        if self.x_test.shape[0]>0:
            self.x_test.tofile(self.file_prefix+"x_test.dat")
            self.y_test.tofile(self.file_prefix+"y_test.dat")
        # print("data file was saved successfully!\n")

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_data_no_label(self):
        return self.no_label_test

    def print_class_info(self):
        for key, val in self.classes_num.items():
            print("class {} has training {}, test {}".format(key,
                                                             val['train_num'],
                                                             val['test_num']))

def main():
    args = get_input_arguments()
    training_mode = args.mode
    data_process = DataPreprocess(conf.n_timestamps, conf.D, conf.step_size,
                                    conf.ntx_max, conf.ntx, conf.nrx_max, 
                                    conf.nrx, conf.nsubcarrier_max, conf.nsubcarrier, 
                                    conf.data_shape_to_nn,
                                    conf.file_prefix,conf.label)
    data_process.load_image(training_mode, True)
    if conf.normalization:
        data_process.normalize_image()
    data_process.signal_processing(conf.do_fft, conf.fft_shape)
    data_process.prepare_shape()
    data_process.save2file()

if __name__ == "__main__":
    main()
