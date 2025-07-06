import numpy as np
def compute_psd_de(data, window, fs, f_bands=None):
    """
    compute  DE (differential entropy) and PSD (power spectral density) features

    input:
	data-[n, m] n channels, m points of each time course,
	window-integer, window lens of each segment in seconds, such as 1s
	fs-integer, frequency of singal sampling rate, such as 200Hz
	optional  f_bands, default delta, theta, aplha, beta, gamma

    output:
        psd,de
    """
    # segment the data
    channels, lens = np.shape(data)
    segment_lens = int(window * fs)
    samples = int(lens // segment_lens)
    data = data[:, :samples * segment_lens]
    data = data.reshape(channels, samples, -1)

    if f_bands == 'meg':
        f_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 100)]  # delta, theta, aplha, beta, gamma
    elif f_bands == 'egg':
        f_bands = [(0, 0.1)]

        # compute the magnitudes
    fxx = np.fft.fft(data)
    timestep = 1 / fs
    f = np.fft.fftfreq(segment_lens, timestep)[:segment_lens // 2]  # only use the positive frequency
    fxx = np.abs(fxx[:, :, :segment_lens // 2])

    psd_bands = []
    de_bands = []
    for f_band1, f_band2 in f_bands:
        f_mask = (f >= f_band1) & (f <= f_band2)
        data_bands = fxx[:, :, f_mask]
        psd = np.mean(data_bands ** 2,axis=-1)
        de = np.log2(2 * np.pi * np.exp(1) * data_bands.var(axis=-1)) / 2

        psd_bands.append(psd)
        de_bands.append(de)
    psd = np.stack(psd_bands)
    de = np.stack(de_bands)
    return psd, de