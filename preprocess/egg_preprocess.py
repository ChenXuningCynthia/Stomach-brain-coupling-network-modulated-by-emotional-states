import numpy as np
from scipy.signal import argrelextrema

def egg_preprocess(rawfile,eggfile):
    ch_name = ['BIO001', 'BIO002', 'BIO003', 'BIO004']
    egg = rawfile.copy().pick(picks=ch_name)

    types={'BIO001':'eeg','BIO002':'eeg','BIO003':'eeg','BIO004':'eeg'}
    egg.set_channel_types(mapping=types)

    egg.resample(sfreq=20)
    method_kw = dict(n_fft=200 * 20, n_overlap=150 * 30)
    egg_psd = egg.compute_psd(method='welch', **method_kw)
    egg_psd.plot()

    find_psd = egg_psd.get_data(fmin=0.03, fmax=0.07)
    greater = argrelextrema(find_psd, np.greater,axis=1)
    index_middle = (find_psd.shape[1] - 1) / 2
    cal = np.abs(greater[1] - index_middle)
    min_channel = np.where(cal == np.min(cal))
    temp = np.zeros(min_channel[0].shape)
    for m in range(min_channel[0].shape[0]):
        temp[m]=find_psd[greater[0][min_channel[0][m]], greater[1][min_channel[0][m]]]
    max=min_channel[0][np.argmax(temp)]
    fhz=0.03+greater[1][max]*0.04/(find_psd.shape[1]-1)
    f_low=fhz-0.015
    f_high=fhz+0.015
    egg.filter(l_freq=f_low, h_freq=f_high, l_trans_bandwidth=f_low*0.15, h_trans_bandwidth=f_high*0.15, method='fir',
               fir_window='hamming')
    egg_channel = greater[0][max] + 1
    cha_name = ['BIO00' + str(egg_channel)]
    psd_egg = egg.copy().pick_channels(cha_name)
    pick_egg = egg.copy().pick_channels(['BIO001'])
    pick_egg._data[0, :] = psd_egg._data
    pick_egg.save(eggfile, tmin=0, tmax=egg.times.max(), overwrite=True)