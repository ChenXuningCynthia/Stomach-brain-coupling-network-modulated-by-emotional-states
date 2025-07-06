import numpy as np
from scipy.signal import hilbert
import hparams as hp
import os.path as op
import mne
import pickle
import pandas as pd

def extract_phase_spectrum(imfs):
    phase_spectrum = imfs.copy()
    i=0
    for imf in phase_spectrum:
        phase = np.angle(hilbert(imf))
        phase_spectrum[i,:]=phase
        i=i+1
    return phase_spectrum

def envelope(imf):
    ana_sig=hilbert(imf)
    env= np.abs(ana_sig)
    return env


def MI(amp,phase):
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)  # Create phase bins
    bin_indices = np.digitize(phase, phase_bins) - 1  # Assign each phase to a bin
    P = np.zeros(n_bins)
    for i in range(n_bins):
        P[i] = np.mean(amp[bin_indices == i])
    P /= np.sum(P)
    H=0
    for m in range(n_bins):
        H=H-P[m]*np.log10(P[m])
    Dkl = np.log10(18) - H
    MI = Dkl / np.log10(18)
    return MI

def time_MI(amp,phase):
    n_time = int((len(amp) - (60 * 400)) / 4000)
    time = np.zeros(n_time + 2)
    all=MI(amp,phase)
    last=MI(amp[-60*400:],phase[-60*400:])
    time[0]=all
    time[-1]=last
    for t in range(n_time):
        time[t+1]=MI(amp[10*400*t:(10*400*t+60*400)],phase[10*400*t:(10*400*t+60*400)])
    return time

sublist = hp.sublist
subnum = hp.subnum
trialnum=hp.trialnum
subjects_pro_meg_dir = hp.subjects_pro_meg_dir
egg_fs=400
meg_fs=400
num_fs=5
n_bins=18
num_voxel = 2052

video_file='video_sequence.xlsx'
video_sequence_all=pd.read_excel(video_file)
video_time=hp.video_time

bands=[
    {'name':'delta','fmin':1,'fmax':4,'index':0},
    {'name':'theta','fmin':4,'fmax':8,'index':1},
    {'name':'alpha','fmin':8,'fmax':14,'index':2},
    {'name':'beta','fmin':14,'fmax':30,'index':3},
    {'name':'gamma','fmin':30,'fmax':100,'index':4},
]

for sb in range(subnum):
    subname = sublist[sb]
    video_sequence=np.array(video_sequence_all)[sb]
    print(sb,subname)
    subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
    subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
    meegfiletag = '-meg-'
    subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                        + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                        + hp.spacing + 'new-ico' + str(hp.ico_downsampling) + '-losse' \
                                        + str(hp.loose))
    mi_all = []
    for run in range(trialnum):
        stc_name = op.join(subjects_stc_dir_sub_this, 'stc_' + str(run))
        stc = mne.read_source_estimate(stc_name)
        subjects_raw_meg_dir_sub = op.join(hp.data_dir, subname)
        rawdir = op.join(subjects_raw_meg_dir_sub, 'run' + str(run) + hp.raw_MEG_name + '.fif')
        rawfile = mne.io.read_raw_fif(rawdir, preload=True)
        eggfile = op.join(subjects_pro_meg_dir_sub, 'run' + str(run) + hp.processed_EGG_name + '.fif')
        egg = mne.io.read_raw_fif(eggfile, preload=True)

        events = mne.find_events(rawfile)
        video_length_1=video_time[video_sequence[(run-2)*2]-1]
        video_length_2 = video_time[video_sequence[(run - 2) * 2+1]-1]

        time = np.zeros(4)
        time[0] = events[0, 0] / 1000 - rawfile._first_time+5
        time[1] = time[0]+video_length_1
        time[2] = time[1] + 50
        time[3] = time[2]+video_length_2
        time = np.round(time, 2)

        egg_close = egg.copy().crop(tmin=time[0], tmax=time[1])
        egg_close_data = egg_close._data
        egg_close_angle = extract_phase_spectrum(egg_close_data)
        egg_open = egg.copy().crop(tmin=time[2], tmax=time[3])
        egg_open_data = egg_open._data
        egg_open_angle = extract_phase_spectrum(egg_open_data)
        # here to continue!!!!

        for fre in bands:
            print(fre['index'], "------------------------------")
            stc_fre=stc.copy()
            stc_fre.data = np.float64(stc_fre.data)
            stc_fre.filter(l_freq=fre['fmin'], h_freq=fre['fmax'],l_trans_bandwidth=0.01, h_trans_bandwidth=0.01)
            target_band = (fre['fmin'], fre['fmax'])
            meg_close = stc_fre.copy().crop(tmin=time[0], tmax=time[1])
            meg_close_data = meg_close.data
            meg_open = stc_fre.copy().crop(tmin=time[2], tmax=time[3])
            meg_open_data = meg_open.data

            for m in range(num_voxel):
                if m == 0:
                    final_meg_data1 = time_MI(envelope(meg_close_data[m, :]), egg_close_angle[0, :])
                    final_data1 = np.zeros([num_voxel, final_meg_data1.shape[0]])
                    final_data1[m, :] = final_meg_data1
                else:
                    final_data1[m, :] = time_MI(envelope(meg_close_data[m, :]), egg_close_angle[0, :])

            for m in range(num_voxel):
                if m == 0:
                    final_meg_data2 = time_MI(envelope(meg_open_data[m, :]), egg_open_angle[0, :])
                    final_data2 = np.zeros([num_voxel, final_meg_data2.shape[0]])
                    final_data2[m, :] = final_meg_data2
                else:
                    final_data2[m, :] = time_MI(envelope(meg_open_data[m, :]), egg_open_angle[0, :])

            if fre['index'] == 0:
                final_MI1 = np.zeros([num_fs, num_voxel, final_meg_data1.shape[0]])
                final_MI1[0, :, :] = final_data1
                final_MI2 = np.zeros([num_fs, num_voxel, final_meg_data2.shape[0]])
                final_MI2[0, :, :] = final_data2
            else:
                final_MI1[fre['index'], :, :] = final_data1
                final_MI2[fre['index'], :, :] = final_data2


        mi_all.append(list(final_MI1))
        mi_all.append(list(final_MI2))

        mat_name = op.join(subjects_stc_dir_sub_this, 'emotion_SBMI.pickle')
        with open(mat_name, 'wb') as f:
            pickle.dump({'mi_all': mi_all}, f)