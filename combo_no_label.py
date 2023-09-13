#!/usr/bin/python3

import numpy as np

import wifi_data_config as conf
from data_learning import NeuralNetworkModel
from data_preprocessing import DataPreprocess
from parse_data_from_log import DataLogParser


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Argument passing the folder to load the training and test data from
    parser.add_argument('-m', '--mixedLabel',
                        help="the mixed label to use for data conversion, defaults to conf.mixed_label_name", type=str,
                        default='None')

    args = parser.parse_args()

    if (args.mixedLabel == 'None'):
        mixedLabel = conf.mixed_label_name
    else:
        mixedLabel = args.mixedLabel

    ##################################################
    # parse data from original data & construct images
    ##################################################

    print("parsing data from log files which are generated by Atheros-CSI-TOOL\n")
    data_generator = DataLogParser(conf.n_timestamps, conf.D, conf.step_size,
                                   conf.ntx_max, conf.nrx_max,
                                   conf.nsubcarrier_max, conf.file_prefix,
                                   conf.log_file_prefix,
                                   conf.skip_frames,
                                   conf.time_offset_ratio,
                                   conf.date_test_conf,
                                   conf.label)

    data_generator.generate_image_no_label(conf.mixed_dates, mixedLabel)
    # train_data, test_data: classes (key: label, value: images under this label)
    test_data = data_generator.get_data_no_label()

    ##################################################
    # apply signal processing blocks to images
    ##################################################

    print("Pre-processing data\n")
    data_process = DataPreprocess(conf.n_timestamps, conf.D, conf.step_size,
                                  conf.ntx_max, conf.ntx, conf.nrx_max,
                                  conf.nrx, conf.nsubcarrier_max, conf.nsubcarrier,
                                  conf.data_shape_to_nn,
                                  conf.file_prefix, conf.label)
    data_process.add_image_no_label(test_data)

    # Naveed: normalization set to False by default
    if conf.normalization:
        data_process.normalize_image()

    # Naveed: FFT'd data (both training and test data)
    data_process.signal_processing(conf.do_fft, conf.fft_shape)
    data_process.prepare_shape()
    final_test_data = data_process.get_data_no_label()

    ##################################################
    # train or test data with neural netowrk
    ##################################################

    nn_model = NeuralNetworkModel(conf.data_shape_to_nn, conf.total_classes)
    print("Get test result using existing model (in test mode)\n")

    # Naveed: a pre-trained motion detection model is being loaded and data is passed through it
    nn_model.load_model(conf.model_name)

    # Naveed: the key variable below assumes to the label/class string such as 'empty', 'apollo' etc.
    for key in final_test_data:
        # new_fig = plt.figure()
        # total_test = len(final_test_data[key])
        # cc = 1
        for idx in final_test_data[key]:
            result = nn_model.get_no_label_result(final_test_data[key][idx])
            result.tofile(conf.file_prefix + key + '_' + idx + '_result.dat')
            # nn_model.save_result(result, 'data/mixed_result/'+key+'_'+idx+m_type+'.dat')

            # Naveed: motion detection results: 0->No motion detected, 1->Motion detected
            print('{} {} number of 0 {} number of 1 {}'.format(key, idx, np.sum(result < 0.5), np.sum(result >= 0.5)))
            # plt.subplot(total_test, 1, cc)
            # plt.plot(result)
            # plt.title(idx)
            # plt.ylim(0, 1.05)
            # cc = cc + 1
        # plt.suptitle(key)
    nn_model.end()
    # plt.show()
    print("Done!")


if __name__ == "__main__":
    main()

