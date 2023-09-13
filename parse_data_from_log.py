#!/usr/bin/python3

import argparse

import numpy as np

import wifi_data_config as conf
from log_parsing import ParseDataFile


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


class ConstructImage:
    def __init__(self, n_timestamps, D, step_size, ntx, nrx, n_tones,skip_frames, offset_ratio):
        self.n_timestamps = n_timestamps
        self.D = D
        self.frame_dur = conf.frame_dur*1e3  # in microseconds
        self.ntx_max = ntx
        self.nrx_max = nrx
        self.n_tones = n_tones
        self.step_size = step_size
        self.skip_frames = skip_frames
        self.time_offset_tolerance = self.n_timestamps*offset_ratio*self.D*self.frame_dur
        print('allowed time offset {}'.format(self.time_offset_tolerance))

    def process_data(self, frame_data, save_filename):
        frame_data = frame_data[self.skip_frames:-self.skip_frames] 
        num_instances = max(0, int((len(frame_data)-self.n_timestamps*self.D)/self.step_size)+5)
        
        final_data = np.zeros((num_instances, self.n_timestamps, self.nrx_max, self.ntx_max, self.n_tones), dtype=np.complex64)
        csi_data = np.zeros((len(frame_data), self.nrx_max, self.ntx_max, self.n_tones), dtype=np.complex64)
        for i in range(len(frame_data)):
            nc, nr = frame_data[i]['format'].nc, frame_data[i]['format'].nr
            csi_data[i, :nr, :nc, :self.n_tones] = frame_data[i]['csi']
        print('parsed csi has shape ', csi_data.shape)
        start_frame_indexes = np.zeros((num_instances,), dtype=np.int32)

        if num_instances == 0:
            return final_data
        d = 0
        valid_instance_c = 0
        time_invalid_c = 0
        rssi = []
        while d<len(frame_data)-self.n_timestamps*self.D:
            temp_image = np.zeros((self.n_timestamps, self.nrx_max, self.ntx_max, self.n_tones), dtype=np.complex64)
            valid = True
            offset = self.step_size
            start_time = 0 
            end_time = 0
            start_index = 0
            end_index = 0
            time_index = []
            for k in range(self.n_timestamps):
                m = d + k*self.D
                nc = frame_data[m]['format'].nc
                csi = frame_data[m]['csi']
                if nc<self.ntx_max:
                    #print("not enough transmit antenna") 
                    valid = False
                    offset = k*self.D+1
                    break
                if k==0:
                    start_time = frame_data[m]['format'].timestamp
                    start_index = m
                elif k==self.n_timestamps-1:
                    end_time = frame_data[m]['format'].timestamp
                    end_index = m
                    time_off = abs(end_time-start_time-(self.n_timestamps-1)*self.D*self.frame_dur)
                    if end_time < start_time: # reseting error, skip
                        #print("end time is smaller than start time")
                        valid = False
                        offset = k*self.D+1
                        break
                    if time_off>self.time_offset_tolerance:
                        #print("timing off is {:.3f}".format(time_off/(self.D*self.frame_dur)))
                        time_invalid_c += 1
                        valid = False
                        offset = 1
                        break
                temp_image[k, :, :nc, :] = csi
                time_index.append(frame_data[m]['format'].timestamp)
            if valid:
                start_frame_indexes[valid_instance_c] = start_index
                rssi.append(frame_data[m]['format'].rssi)
                final_data[valid_instance_c, ...] = temp_image
                valid_instance_c = valid_instance_c+1
                #v = check_no_outlier(temp_image)
            d = d+offset
        final_data = final_data[:valid_instance_c,...]
        start_frame_indexes = start_frame_indexes[:valid_instance_c]
        csi_data.tofile(save_filename+'_raw.dat')
        print('saved parsed CSI to ', save_filename+'_raw.dat')
        start_frame_indexes.tofile(save_filename+'_start_frame.dat')
        print('saved start frame indexes to ', save_filename+'_start_frame.dat')
        print("total number of valid images {}, number of images has duration too large {} ".format(valid_instance_c, time_invalid_c))
        print('total number of valid start indexes {}'.format(start_frame_indexes.shape[0]))
        return final_data


class DataLogParser:
    def __init__(self, n_timestamps, D, step_size, ntx_max,
                 nrx_max, nsubcarrier_max, file_prefix, log_file_prefix, skip_frames, time_offset_ratio,conf, labels):
        self.parser = ParseDataFile()
        self.image_constructor = ConstructImage(n_timestamps, D, step_size,
                                                ntx_max, nrx_max, nsubcarrier_max, skip_frames, time_offset_ratio)
        self.file_prefix = file_prefix
        self.log_file_prefix = log_file_prefix
        self.classes = {}
        self.data_shape = (n_timestamps, nrx_max, ntx_max, nsubcarrier_max)
        self.step_size = step_size
        self.n_timestamps = n_timestamps
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_evaluate = {}
        self.classes_num = {}
        self.conf = conf
        self.out_data_train = {}
        self.out_data_test = {}
        self.out_data_no_label = {}
        self.label = labels
        for k, o in self.label:
            self.out_data_train[o] = np.array([])
            self.out_data_test[o] = np.array([])

    def generate_image(self, classes, date):
        for idx, m in enumerate(date):
            #print("\nprocessing data from date {}\n".format(m))
            #print(self.conf[m])
            filename = self.file_prefix + m + '/'
            logfilename = self.log_file_prefix + m + '/'
            for k,o in self.label:
                out_data_train = np.array([])
                out_data_test = np.array([])
                for ii in range(len(classes[k]['label name'])): 
                    label_name = classes[k]['label name'][ii]
                    if label_name in self.conf[m]:
                        total_tests = self.conf[m][label_name]
                    else:
                        continue
                    if total_tests == 0:
                        continue
                    if classes[k]['test'][idx][ii] is None:
                        raise ValueError('validation test num should not be None!')
                    a = np.arange(1, total_tests+1)
                    np.random.shuffle(a)
                    test_index = a[:classes[k]['test'][idx][ii]]
                    if len(test_index) == 0:
                        print(logfilename + label_name+' all used as train\n')
                    elif len(test_index) == total_tests:
                        print(logfilename + label_name+' all used as test\n')
                    else:
                        print("test index: {}".format(test_index))

                    for i in range(1, total_tests+1):
                        print('parsing: ', logfilename + label_name + str(i) + ".data")
                        frame_data = self.parser.parse(logfilename + label_name + str(i) + ".data", self.conf[m]['p'])
                        dd = self.image_constructor.process_data(frame_data, self.file_prefix+m+'_'+label_name+'_'+str(i))
                        if i in test_index:
                            self.out_data_test[o] = append_array(self.out_data_test[o], dd)
                        else:
                            self.out_data_train[o] = append_array(self.out_data_train[o], dd)

    def generate_image_no_label(self, date, label_name):
        for idx, m in enumerate(date):
            #print("\nprocessing data from date {}\n".format(m))
            #print(self.conf[m])
            filename = self.file_prefix + m + '/'
            logfilename = self.log_file_prefix + m + '/'
            if label_name not in self.conf[m]:
                continue
            total_tests = self.conf[m][label_name]
            if total_tests == 0:
                continue
            self.out_data_no_label[m] = {}
            print('on date '+m)
            for i in range(1, total_tests+1):
                frame_data = self.parser.parse(logfilename + label_name + str(i) + ".data", self.conf[m]['p'])
                dd = self.image_constructor.process_data(frame_data, self.file_prefix+m+'_'+label_name+'_'+str(i))
                self.out_data_no_label[m][label_name+'_'+str(i)] = dd 
                print('add mixed data: '+label_name+'_'+str(i))

    def save_data(self, train_model):
        for k, o in self.label:
            if train_model:
                self.out_data_train[o].tofile(self.file_prefix+"training_"+str(o)+'.dat')
                self.out_data_test[o].tofile(self.file_prefix+"training_test_"+str(o)+'.dat')
            else:
                self.out_data_test[o].tofile(self.file_prefix+"test_"+str(o)+'.dat')

    def get_data(self):
        return self.out_data_train, self.out_data_test

    def get_data_no_label(self):
        return self.out_data_no_label

def main():
    args = get_input_arguments()
    training_mode = args.mode
    if training_mode:
        classes = conf.training_classes
        date = conf.training_date
    else:
        classes = conf.test_classes
        date = conf.test_date
    data_generator = DataLogParser(conf.n_timestamps, conf.D, conf.step_size,
                                    conf.ntx_max, conf.nrx_max, 
                                    conf.nsubcarrier_max, conf.file_prefix,
                                    conf.log_file_prefix,
                                    conf.skip_frames,
                                    conf.time_offset_ratio,
                                    conf.date_test_conf,
                                    conf.label)
    data_generator.generate_image(classes, date)
    data_generator.save_data(training_mode)

if __name__ == "__main__":
    main()
