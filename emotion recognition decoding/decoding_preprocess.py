import numpy as np
import hparams as hp
import os.path as op
import mne
import scipy.io as io
import pickle
import pandas as pd
from psd_de import compute_psd_de

if __name__ == '__main__':
    sublist = hp.sublist
    subjects_pro_meg_dir = hp.subjects_pro_meg_dir
    video_file='video_sequence.xlsx'
    video_sequence_all=pd.read_excel(video_file)
    video_time=hp.video_time

    for sb in range(0,16):
        subname = sublist[sb]
        print(sb,subname)
        video_sequence = np.array(video_sequence_all)[sb]
        subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
        subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
        meegfiletag = '-meg-'
        subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                            + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                            + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                            + str(hp.loose))
        meg_all_psd = []
        meg_all_de = []
        egg_all_psd = []
        egg_all_de = []

        for run in range(2, 8):
            print(run, 'run')

            subjects_raw_meg_dir_sub = op.join(hp.data_dir, subname)
            rawdir = op.join(subjects_raw_meg_dir_sub, 'run' + str(run) + hp.raw_MEG_name + '.fif')
            rawfile = mne.io.read_raw_fif(rawdir, preload=True)
            eggdir=op.join(subjects_pro_meg_dir_sub,'run'+ str(run) +'_fs20_egg_tsss.fif')
            eggfile = mne.io.read_raw_fif(eggdir, preload=True)
            meegfiletag = '-meg-'
            subjects_stc_sub = op.join(hp.subjects_stc_dir, subname, hp.processed_MEG_name, hp.stc_method \
                                       + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                       + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                       + str(hp.loose))
            result_name = op.join(hp.project_path, 'result', subname, str(run) + '.mat')
            result_emo = io.loadmat(result_name)['result']

            events = mne.find_events(rawfile)
            video_length_1 = video_time[video_sequence[(run - 2) * 2] - 1]
            video_length_2 = video_time[video_sequence[(run - 2) * 2 + 1] - 1]

            time = np.zeros(4)
            time[0] = events[0, 0] / 1000 - rawfile._first_time + 5
            time[1] = time[0] + video_length_1
            time[2] = time[1] + 50
            time[3] = time[2] + video_length_2
            time = np.round(time, 2)

            stc_fname = op.join(subjects_stc_sub, 'raw_2017_' + str(run))
            stc = mne.read_source_estimate(stc_fname)
            meg_sfreq = 400
            egg_sfreq = 20
            winlength = 20
            for m in range(2):
                tmin = int(time[2 * m] * meg_sfreq)
                tmax = int(time[2 * m + 1] * meg_sfreq)
                meg_data = stc.data[:, tmin:tmax]
                egg_data=eggfile.copy().crop(tmin=time[2 * m],tmax=time[2 * m + 1])._data
                meg_psd, meg_de = compute_psd_de(meg_data, winlength, meg_sfreq)
                egg_psd, egg_de = compute_psd_de(egg_data, winlength, egg_sfreq)

                meg_all_psd.append(meg_psd)
                meg_all_de.append(meg_de)
                egg_all_psd.append(egg_psd)
                egg_all_de.append(egg_de)

        meg_save_dir = op.join(subjects_stc_dir_sub_this, 'emotion_MEG_EGG_DE_PSD.pickle')
        with open(meg_save_dir, 'wb') as f:
            pickle.dump({'meg_psd':meg_all_psd,'egg_psd':egg_all_psd,'meg_de':meg_all_de,'egg_de':egg_all_de}, f)


