import wifi_data_config as conf
from functools import partial
import numpy as np


def fft_func(data, fft_shape, num_dims, fft_axes=(0,), fftshift=True, log_norm=True):
    temp_data = data
    if num_dims == 1:
        temp_data = np.abs(np.fft.fft(temp_data, n=fft_shape[0], axis=fft_axes[0]))
        if fftshift:
            temp_data = np.fft.fftshift(temp_data, fft_axes=(fft_axes[0],))
    else:
        temp_data = np.abs(np.fft.fft2(temp_data, s=fft_shape, axes=fft_axes))
        if fftshift:
            temp_data = np.fft.fftshift(temp_data, axes=fft_axes)
    if log_norm:
        temp_data = np.log10(temp_data + 1)
    return temp_data


def get_central_crop(data, crop_window):
    lft_axis_0 = data.shape[1] // 2 - crop_window[0] // 2
    rgt_axis_0 = data.shape[1] // 2 + crop_window[0] // 2

    lft_axis_1 = data.shape[2] // 2 - crop_window[1] // 2
    rgt_axis_1 = data.shape[2] // 2 + crop_window[1] // 2
    cropped = data[:,
              lft_axis_0:rgt_axis_0,
              lft_axis_1:rgt_axis_1 + 1,
              ...]
    return cropped


def get_preprocessed_data(raw_data):
    raw_data = np.transpose(raw_data, [0, 1, 4, 3, 2])  # re-arrange axes for clarity
    raw_data = np.reshape(raw_data, raw_data.shape[:3] + (-1,))  # flatten tx, tr dimensions
    raw_data /= raw_data[:, :1, ...]  # normalize by first CSI frame
    amplitude = abs(raw_data)  # take magnitude

    ftd = fft_2d_only(data=amplitude, fft_shape=(128, 14), fft_axes=(1, 2))  # 2D DFT
    cropped = get_central_crop(ftd, (8, 3))  # central crop
    averaged = np.mean(cropped, axis=-1)  # averaging
    averaged = averaged.reshape((averaged.shape[0], -1))  # flattening

    return averaged


def get_train_data_from_file(file_prefix, posfix, data_type, label_type, data_shape):
    x_train = np.fromfile(file_prefix + 'x_train' + posfix + '.dat', dtype=data_type)
    x_test = np.fromfile(file_prefix + 'x_test' + posfix + '.dat', dtype=data_type)
    x_train = np.reshape(x_train, (-1,) + data_shape)
    x_test = np.reshape(x_test, (-1,) + data_shape)

    y_train = np.fromfile(file_prefix + 'y_train' + posfix + '.dat', dtype=label_type)
    y_test = np.fromfile(file_prefix + 'y_test' + posfix + '.dat', dtype=label_type)

    print('train ds size ', x_train.shape, y_train.shape)
    unique, unique_counts = np.unique(y_train, return_counts=True)
    print('label counts | train : ', unique, unique_counts)

    print('test ds size ', x_test.shape, y_test.shape)
    unique, unique_counts = np.unique(y_test, return_counts=True)
    print('label counts | test : ', unique, unique_counts)

    return x_train, y_train, x_test, y_test


def write_to_files(x_train, y_train, x_test, y_test, postfix="preprocessed", file_prefix=conf.file_prefix):
    if x_train.shape[0]:
        x_train.tofile(file_prefix + f'x_train{postfix}.dat')
        y_train.tofile(file_prefix + f'y_train{postfix}.dat')

    if x_test.shape[0]:
        x_test.tofile(file_prefix + f'x_test{postfix}.dat')
        y_test.tofile(file_prefix + f'y_test{postfix}.dat')


if __name__ == "__main__":
    fft_2d_only = partial(fft_func, num_dims=2)
    # load CSI arrays
    x_train, y_train, x_test, y_test = get_train_data_from_file(conf.file_prefix, '_csi', np.complex64, np.int8,
                                                                (conf.win_len, conf.ntx, conf.nrx, conf.nsubcarrier))

    if x_train.shape[0]:
        x_train = get_preprocessed_data(x_train)
    if x_test.shape[0]:
        x_test = get_preprocessed_data(x_test)
    write_to_files(x_train, y_train, x_test, y_test, postfix="")
