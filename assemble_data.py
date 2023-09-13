#!/usr/bin/env python3

import shutil
import numpy as np
import os
import wifi_data_config as conf
from train_test_config import *

def append_array(array_a, array_b, axis=0):
    if array_b.size == 0:
        return array_a
    if array_a.size == 0:
        array_a = array_a.astype(array_b.dtype)
        array_a = array_a.reshape((0,) + array_b.shape[1:])
    array_a = np.concatenate([array_a, array_b], axis)
    return array_a


def data_from_file(labels, date_conf, train_date, test_date, exclude_date, window_len, file_prefix):
    train_final_data = {}
    test_final_data = {}
    for m in labels:
        train_final_data[m] = np.array([], dtype=np.complex64)
        test_final_data[m] = np.array([], dtype=np.complex64)
        for date in date_conf:
            if date not in train_date and date not in test_date:
                continue
            for d_type in labels[m]:
                if d_type in date_conf[date]:
                    total_test = date_conf[date][d_type]
                    for idx in range(1, total_test + 1):
                        valid = True
                        for ex_d, ex_t, ex_i in exclude_date:
                            if date == ex_d and d_type == ex_t and idx == ex_i:
                                valid = False
                                print('exclude ', ex_d, ex_t, ex_i)
                                break
                        if not valid: continue

                        this_prefix = file_prefix + date + '_' + d_type + '_' + str(idx)
                        raw_csi_filename = this_prefix + '_raw.dat'
                        raw_csi = np.fromfile(raw_csi_filename, dtype=np.complex64)
                        raw_csi = np.reshape(raw_csi, (-1, 3, 3, 56))
                        start_frame_filename = this_prefix + '_start_frame.dat'
                        start_frame = np.fromfile(start_frame_filename, dtype=np.int32)
                        result_frame_filename = this_prefix + '_result.dat'
                        result_frame = np.fromfile(result_frame_filename, dtype=np.float32)
                        if m == 0:
                            detected_start_frame = start_frame[result_frame < 0.5]
                            # result_frame = result_frame[result_frame<0.5]
                        else:
                            detected_start_frame = start_frame[result_frame >= 0.5]
                            # result_frame = result_frame[result_frame>=0.5]
                        result_frame = result_frame.tolist()
                        start_frame = start_frame.tolist()

                        chosen_csi = np.zeros(shape=(len(detected_start_frame), window_len, 3, 3, 56),
                                              dtype=np.complex64)
                        cnt = 0
                        ignore_cnt = 0
                        all_detected_cnt = []
                        for i in range(len(start_frame)):
                            if m == 0 and result_frame[i] >= 0.5: continue
                            if m != 0 and result_frame[i] < 0.5: continue

                            start_idx, end_idx = start_frame[i], start_frame[i] + window_len
                            if end_idx >= len(raw_csi): break

                            j, detected_cnt = i, 0
                            while start_frame[j] <= end_idx - conf.fft_shape[0]:
                                if m == 0 and result_frame[j] < 0.5: detected_cnt += 1
                                if m != 0 and result_frame[j] >= 0.5: detected_cnt += 1
                                j += 1
                                if j >= len(start_frame): break

                            if detected_cnt == 0:
                                print(
                                    'something is wrong, should not detecting nothing! frame start from {} to {}'.format(
                                        start_idx, end_idx))
                            all_detected_cnt.append(detected_cnt)
                            if detected_cnt < conf.win_detect_cnt:
                                ignore_cnt += 1
                                continue
                            chosen_csi[cnt, ...] = raw_csi[start_idx:end_idx, ...]
                            cnt += 1
                        chosen_csi = chosen_csi[:cnt, ..., :]
                        # print('result frame {} chosen_csi {}'.format(len(result_frame), len(chosen_csi)))
                        print('total cnt {} ignore cnt {} valid cnt {}'.format(len(all_detected_cnt), ignore_cnt, cnt))
                        print('detected cnt in a window min {} max {} median {}'.format(np.min(all_detected_cnt),
                                                                                        np.max(all_detected_cnt),
                                                                                        np.median(all_detected_cnt)))

                        if date in test_date:
                            print('to test')
                            print('date {} type {} idx {} add from {} to {}'.format(date, d_type, idx,
                                                                                    test_final_data[m].shape[0],
                                                                                    test_final_data[m].shape[0] +
                                                                                    chosen_csi.shape[0]))
                            test_final_data[m] = append_array(test_final_data[m], chosen_csi)
                        elif date in train_date:
                            print('to training')
                            print('date {} type {} idx {} add from {} to {}'.format(date, d_type, idx,
                                                                                    train_final_data[m].shape[0],
                                                                                    train_final_data[m].shape[0] +
                                                                                    chosen_csi.shape[0]))
                            train_final_data[m] = append_array(train_final_data[m], chosen_csi)

    print('training data: ')
    x_train = np.array([], dtype=np.complex64)
    y_train = np.array([], dtype=np.int8)
    for m in train_final_data:
        temp_label = np.zeros((train_final_data[m].shape[0],), dtype=np.int8)
        temp_label[:] = m
        permute_idx = np.random.permutation(train_final_data[m].shape[0])
        # train_final_data[m] = train_final_data[m][permute_idx[:4000], ...]
        # only save 14 (56/4) evenly spaced subcarriers to save space
        x_train = append_array(x_train, train_final_data[m][..., ::4])
        y_train = append_array(y_train, temp_label)
        print('label {} has shape {}'.format(m, train_final_data[m].shape))
    unique, unique_counts = np.unique(y_train, return_counts=True)
    print('checking label ', unique, unique_counts)

    print('x_train shape ', x_train.shape)
    x_train.tofile(file_prefix + 'x_train.dat')
    y_train.tofile(file_prefix + 'y_train.dat')

    print('test data: ')
    x_test = np.array([], dtype=np.complex64)
    y_test = np.array([], dtype=np.int8)
    for m in test_final_data:
        temp_label = np.zeros((test_final_data[m].shape[0],), dtype=np.int8)
        temp_label[:] = m
        permute_idx = np.random.permutation(test_final_data[m].shape[0])
        # test_final_data[m] = test_final_data[m][permute_idx[:4000], ...]
        # only save 14 (56/4) evenly spaced subcarriers to save space
        x_test = append_array(x_test, test_final_data[m][..., ::4])
        y_test = append_array(y_test, temp_label)
        print('label {} has shape {}'.format(m, test_final_data[m].shape))
    unique, unique_counts = np.unique(y_test, return_counts=True)
    print('checking label ', unique, unique_counts)

    print('x_test shape', x_test.shape)
    x_test.tofile(file_prefix + 'x_test.dat')
    y_test.tofile(file_prefix + 'y_test.dat')

    return train_final_data, test_final_data


def main():

    date_label = {0: ['empty'],
        1: ['human', ],
        2: ['hazel', 'apollo', 'lincoln_hazel', ]}
    total_classes = 3
    window_len = conf.win_len
    for (dates, exclude_dates, split_type) in ([dates_train, exclude_dates_train, "train"],
                                               [dates_test, exclude_dates_test, "test"]):
        if split_type == "train":
            data_from_file(date_label, conf.date_test_conf, dates, [], exclude_dates,
                           window_len, conf.file_prefix)
            # renaming helps with
            os.rename(conf.file_prefix + "x_train.dat", conf.file_prefix + "x_train_csi.dat")
            os.rename(conf.file_prefix+"y_train.dat", conf.file_prefix+"y_train_csi.dat")
        elif split_type == "test":
            data_from_file(date_label, conf.date_test_conf, [], dates, exclude_dates,
                           window_len, conf.file_prefix)
            os.rename(conf.file_prefix + "x_test.dat", conf.file_prefix + "x_test_csi.dat")
            os.rename(conf.file_prefix + "y_test.dat", conf.file_prefix + "y_test_csi.dat")




if __name__ == "__main__":
    main()
