import numpy as np
from mne.preprocessing import ICA

def meg_preprocess(rawfile,megfile):
    """
    对meg数据进行预处理
    """
    meg = rawfile.copy().pick(picks=['meg', 'bio'])
    meg.notch_filter(freqs=np.arange(50, 500 / 2 + 1, 50))  # 工频滤波
    meg.resample(sfreq=400)  # 降采样,mne has a low filter before resample to avoid anti-aliasing artifacts
    meg.filter(l_freq=1, h_freq=200,l_trans_bandwidth=0.01,h_trans_bandwidth=0.01)  # 滤波
    meg.plot(duration=100)
    meg.plot_psd()

    # ICA to remove eye blink and ECG
    ica_meg = ICA(max_iter='auto', random_state=10)
    ica_meg.fit(meg)
    ica_meg.plot_sources(meg,show_scrollbars=True)
    ica_meg.plot_components()
    ecg_indices, ecg_scores = ica_meg.find_bads_ecg(meg, ch_name='BIO006', method='correlation')
    eog_indices, eog_scores = ica_meg.find_bads_eog(meg, ch_name='BIO007',threshold='auto')
    auto=list(set(eog_indices + ecg_indices))

    ica_meg.exclude = auto
    ica_meg.apply(meg)
    meg_pre=meg.pick(picks=['meg'])
    meg_pre.save(megfile, tmin=0, tmax=meg.times.max(), overwrite=True)