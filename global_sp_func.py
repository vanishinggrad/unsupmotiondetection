import matplotlib.pyplot as plt
import numpy as np


def check_no_outlier(d):
    ampl = np.abs(d)
    ampl = np.sum(ampl**2, axis=-1)
    ampl = ampl[:, :, 0]
    min_g = 0
    for m in range(ampl.shape[-1]):
        temp_ampl = np.sqrt(ampl[:, m])
        temp_diff = np.diff(temp_ampl, n=1)
        temp_diff = temp_diff/temp_ampl[0:-1]
        temp_diff = np.abs(temp_diff) 
        temp_diff = np.sort(temp_diff)
        most_grad = np.quantile(temp_diff, 0.8)
        #min_g = temp_diff[-1] - temp_diff[-5]
        min_g = (temp_diff[-1]-most_grad)/most_grad 
        if min_g>4.0:
            plt.figure()
            plt.plot(temp_ampl)
            plt.show()
            return False
    return True

def reshape_func(d, subcarrier_spacing):
    d = d[..., ::subcarrier_spacing]
    d = np.transpose(d, [0,1, 4, 3,2])
    d = d.reshape(d.shape[:-2]+(-1,))
    return d

def shape_conversion(d, l):
    temp_d = d[:, int(d.shape[1]/2-l/2):int(d.shape[1]/2+l/2), ...] 
    d = temp_d
    return d

def fft_func(data, fft_shape, num_dims):
    temp_data = data
    if num_dims==1:
        temp_data = np.abs(np.fft.fft(temp_data, n=fft_shape[0], axis=1))
        #temp_data /= np.sum(temp_data, axis=(1,), keepdims=True)
        temp_data = np.fft.fftshift(temp_data, axes=(1,))
    else:
        temp_data = np.abs(np.fft.fft2(temp_data, s=fft_shape, axes=(1,2)))
        #temp_data /= np.sum(temp_data, axis=(1,2), keepdims=True)
        temp_data = np.fft.fftshift(temp_data, axes=(1,2))
    temp_data = np.log10(temp_data+1)
    return temp_data

def obtain_angle(symbol_data):
    angle_data = np.zeros(symbol_data.shape[:-1]+(symbol_data.shape[-1]-3,)) 
    for i in range(3):
        diff_data = symbol_data[...,i*3+1:i*3+3]/symbol_data[..., i*3:i*3+1]
        angle_data[..., 2*i:2*i+2] = np.angle(diff_data)
    return angle_data

def iq_split(d):
    d = d/d[:, :1, ...]
    real_part = np.real(d)
    real_part = real_part.astype(np.float32)
    imag_part = np.imag(d)
    imag_part = imag_part.astype(np.float32)
    out = np.concatenate([real_part, imag_part], axis=-1)
    return out


def sp_func(d, do_fft, fft_shape):
    phase = obtain_angle(np.copy(d))
    phase = phase.astype(np.float32)
    ampl = np.abs(d)
    ampl = ampl/ampl[:, :1, ...]
    ampl = ampl.astype(np.float32)
    total_instance = phase.shape[0]
    if do_fft:
        out = np.zeros(((ampl.shape[0],)+fft_shape+(ampl.shape[-1], 2)), dtype=np.float32)
        for i in range(0, total_instance, 5000):
            num = min(total_instance-i, 5000)
            # ampl fft
            out[i:i+num, ..., 0] = fft_func(np.copy(ampl[i:i+num, ...]), fft_shape, 2)
            # phase fft
            unwrap_phase = np.unwrap(phase[i:i+num, ...], axis=1)
            out[i:i+num, ..., :unwrap_phase.shape[-1], 1] = fft_func(unwrap_phase, fft_shape, 1) 
        return out
    else:
        print('no fft!!!')
        out = np.zeros((ampl.shape+(2,)), dtype=np.float32)
        out[..., 0] = ampl
        # phase unwrapping
        for i in range(0, total_instance, 5000):
            num = min(total_instance-i, 5000)
            unwrap_phase = np.unwrap(phase[i:i+num, ...],axis=1)
            out[i:i+num,...,:unwrap_phase.shape[-1], 1] = unwrap_phase
        return out

