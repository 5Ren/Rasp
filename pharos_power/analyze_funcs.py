import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def plot_fft(signal, time_data, ir_or_green):
    sample_numbers = signal.shape[0]

    # fft
    # ---------------------------------------------------------
    large_f = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(sample_numbers, d=time_data)

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))
    fig.suptitle(f'FFT {ir_or_green}')
    ax[0].plot(large_f.real, label="Real part")
    ax[0].legend()
    ax[1].plot(large_f.imag, label="Imaginary part")
    ax[1].legend()
    ax[2].plot(fft_freq, label="Frequency")
    ax[2].legend()
    ax[2].set_xlabel("time [sec]")

    ax[1].axhline(y=0.5, color='r')
    ax[1].axhline(y=-0.5, color='r')
    plt.show()

    Amp = np.abs(large_f / (sample_numbers / 2))  # 振幅

    fig2, ax = plt.subplots()
    ax.set_title(f'Amplitude of the original signal {ir_or_green}')
    ax.plot(fft_freq[1:int(sample_numbers / 2)], Amp[1:int(sample_numbers / 2)])
    ax.set_xlabel("Freqency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.axhline(y=0.0001, color='r')
    plt.show()
    # ---------------------------------------------------------


def lowpass_filter(signal, cutoff_frequency, sample_rate):
    nyquist_frequency = sample_rate / 2
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    print(f'{normalized_cutoff=}')

    # ローパスフィルタの係数を計算する
    b, a = butter(4, normalized_cutoff, btype='low', analog=False)

    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def moving_average(signal, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(signal, window, 'same')


def remove_random_noize(signal, times, random_range):

    signal_numbers = signal.shape[0]
    measurement_signal = np.zeros((times, signal_numbers))
    for i in range(times):
        rand_signal = (np.random.rand(signal_numbers) - 0.5) * random_range
        measurement_signal[i, :] = signal + rand_signal

    return np.mean(measurement_signal, axis=0)




