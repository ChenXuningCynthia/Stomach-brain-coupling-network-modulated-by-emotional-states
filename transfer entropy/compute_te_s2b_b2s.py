import numpy as np
from scipy.signal import hilbert
import hparams as hp
import os.path as op
import mne
import random
import scipy.io as io
import pickle
from infodynamics.demos.python.example9TeKraskovAutoEmbedding import te_with_Ragwitz_criteria
from scipy.signal import resample
from parc import vertex_liner_ridge
import pandas as pd

subnum = hp.subnum
trialnum=hp.trialnum
sublist = hp.sublist
subjects_pro_meg_dir = hp.subjects_pro_meg_dir
egg_fs=20
meg_fs=400
meg_power_fs=400
num_fs=5
num_voxel = 2052

random.seed(36)
bands=[
    {'name':'delta','fmin':1,'fmax':4,'index':0},
    {'name':'theta','fmin':4,'fmax':8,'index':1},
    {'name':'alpha','fmin':8,'fmax':14,'index':2},
    {'name':'beta','fmin':14,'fmax':30,'index':3},
    {'name':'gamma','fmin':30,'fmax':100,'index':4},
]
video_file='video_sequence.xlsx'
video_sequence_all=pd.read_excel(video_file)
video_time=hp.video_time
def envelope(imf):
    ana_sig=hilbert(imf)
    env= np.square(np.abs(ana_sig))
    env=resample(env,int(len(env)/20))
    return env

def clip_te(meg,egg,mode):
    time_clip=60*20
    if len(meg)>len(egg):
        meg=meg[:len(egg)]
    elif len(meg)<len(egg):
        egg=egg[:len(meg)]
    if mode=='all':
        n_time=int(len(meg)/time_clip)
        te_b2s=np.zeros([n_time])
        te_s2b = np.zeros([n_time])
        for i in range(n_time):
            te_b2s[i]= te_with_Ragwitz_criteria(meg[i*time_clip:(i+1)*time_clip],egg[i*time_clip:(i+1)*time_clip])
            te_s2b[i] = te_with_Ragwitz_criteria(egg[i * time_clip:(i + 1) * time_clip],meg[i * time_clip:(i + 1) * time_clip])
        te_B2S=np.mean(te_b2s)
        te_S2B=np.mean(te_s2b)
    if mode=='first':
        te_B2S= te_with_Ragwitz_criteria(meg[:time_clip],egg[:time_clip])
        te_S2B = te_with_Ragwitz_criteria(egg[:time_clip],meg[:time_clip])
    if mode=='last':
        te_B2S= te_with_Ragwitz_criteria(meg[-time_clip:],egg[-time_clip:])
        te_S2B = te_with_Ragwitz_criteria(egg[-time_clip:],meg[-time_clip:])
    return [te_B2S,te_S2B]


for sb in range(subnum):
    subname = sublist[sb]
    print(sb,subname)
    subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
    subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
    meegfiletag = '-meg-'
    subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                        + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                        + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                        + str(hp.loose))
    result_all = np.zeros(12)
    te_all = []
    subjects_meg_dir_sub = op.join(hp.subjects_pro_meg_dir, subname)
    video_sequence = np.array(video_sequence_all)[sb]
    for run in range(trialnum):
        result_name = op.join(hp.project_path, 'result', subname, str(run) + '.mat')
        result = io.loadmat(result_name)['result']
        result_all[(run - 2) * 2] = result[0, 0]
        result_all[(run - 2) * 2 + 1] = result[1, 0]

        stc_name = op.join(subjects_stc_dir_sub_this, 'raw_2017_' + str(run))
        stc = mne.read_source_estimate(stc_name)
        subjects_raw_meg_dir_sub = op.join(hp.data_dir, subname)
        rawdir = op.join(subjects_raw_meg_dir_sub, 'run' + str(run) + hp.raw_MEG_name + '.fif')
        rawfile = mne.io.read_raw_fif(rawdir, preload=True)
        src_fname = op.join(subjects_meg_dir_sub, 'all_tsss' + str(run) + '-epo-meg-oct5-ico-4-src.fif')
        src = mne.read_source_spaces(src_fname)

        eggfile = op.join(subjects_pro_meg_dir_sub, 'run' + str(run) + hp.processed_EGG_name + '.fif')
        egg = mne.io.read_raw_fif(eggfile, preload=True)

        events = mne.find_events(rawfile)
        video_length_1 = video_time[video_sequence[(run - 2) * 2] - 1]
        video_length_2 = video_time[video_sequence[(run - 2) * 2 + 1] - 1]

        time = np.zeros(4)
        time[0] = events[0, 0] / 1000 - rawfile._first_time + 5
        time[1] = time[0] + video_length_1
        time[2] = time[1] + 50
        time[3] = time[2] + video_length_2
        time = np.round(time, 2)

        egg_close = egg.copy().crop(tmin=time[0], tmax=time[1])
        egg_close_data = egg_close._data
        egg_open = egg.copy().crop(tmin=time[2], tmax=time[3])
        egg_open_data = egg_open._data

        te1=np.zeros([num_fs,num_voxel,3,2])
        te2 = np.zeros([num_fs, num_voxel, 3,2])
        for fre in bands:
            stc_fre=stc.copy()
            stc_fre.data = np.float64(stc_fre.data)
            stc_fre.filter(l_freq=fre['fmin'], h_freq=fre['fmax'])
            meg_fre_data = vertex_liner_ridge(subname, stc_fre, src)
            meg_close = stc_fre.copy().crop(tmin=time[0], tmax=time[1])
            meg_close_data = meg_close.data
            meg_open = stc_fre.copy().crop(tmin=time[2], tmax=time[3])
            meg_open_data = meg_open.data


            print(fre['index'],"fs")
            for m in range(num_voxel):
                te1[fre['index'], m, 0,:] = clip_te(envelope(meg_close_data[m, :]), egg_close_data[0, :],mode='all')
                te1[fre['index'], m, 1,:] = clip_te(envelope(meg_close_data[m, :]), egg_close_data[0, :], mode='first')
                te1[fre['index'], m, 2,:] = clip_te(envelope(meg_close_data[m, :]), egg_close_data[0, :], mode='last')

                te2[fre['index'], m, 0,:] = clip_te(envelope(meg_open_data[m, :]), egg_open_data[0, :], mode='all')
                te2[fre['index'], m, 1,:] = clip_te(envelope(meg_open_data[m, :]), egg_open_data[0, :], mode='first')
                te2[fre['index'], m, 2,:] = clip_te(envelope(meg_open_data[m, :]), egg_open_data[0, :], mode='last')

        te_all.append(te1)
        te_all.append(te2)

    happy =[]
    neutral =[]
    sad = []
    fear = []
    for emo in range(12):
        if result_all[emo] == 1:
            happy.append(te_all[emo])
        if result_all[emo] == 2:
            neutral.append(te_all[emo])
        if result_all[emo] == 3:
            sad.append(te_all[emo])
        if result_all[emo] == 4:
            fear.append(te_all[emo])

    mat_name = op.join(subjects_stc_dir_sub_this, 'te_3modes.pickle')
    with open(mat_name, 'wb') as f:
        pickle.dump({'happy': happy,'neutral': neutral,'sad': sad,'fear': fear}, f)

    print(" ")


