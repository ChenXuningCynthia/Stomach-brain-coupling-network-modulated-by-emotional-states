import hparams as hp
import os.path as op
import mne
import scipy.io as io
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

numsub = hp.subnum
trialnum=hp.trialnum
subjects_dir = hp.subjects_mri_dir
src_fname = op.join(subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif")
src_to = mne.read_source_spaces(src_fname)
sublist = hp.sublist
subjects_pro_meg_dir = hp.subjects_pro_meg_dir

egg_fs = 20
meg_fs = 400
meg_power_fs = 200
num_fs = 100
num_voxel = 2052
num_time = 12
tmin = 0
tstep = 0.0025
tmax = tmin + tstep * (num_time - 1)

happy_valence = []
happy_arousal = []
neutral_valence = []
neutral_arousal = []
sad_valence = []
sad_arousal = []
fear_valence = []
fear_arousal = []

def va_distribution():
    for sb in range(numsub):
        subname = sublist[sb]
        happy_v = []
        neutral_v = []
        sad_v = []
        fear_v = []

        happy_a = []
        neutral_a = []
        sad_a = []
        fear_a = []
        for run in range(trialnum):
            print(run, 'run')
            result_name = op.join(hp.project_path, 'result', subname, str(run) + '.mat')
            result = io.loadmat(result_name)['result']

            for emo in range(2):
                if result[emo, 0] == 1:
                    happy_v.append(result[emo, 1])
                    happy_a.append(result[emo, 2])
                if result[emo, 0] == 2:
                    neutral_v.append(result[emo, 1])
                    neutral_a.append(result[emo, 2])
                if result[emo, 0] == 3:
                    sad_v.append(result[emo, 1])
                    sad_a.append(result[emo, 2])
                if result[emo, 0] == 4:
                    fear_v.append(result[emo, 1])
                    fear_a.append(result[emo, 2])

        t_v = stats.zscore(np.concatenate((happy_v, neutral_v, sad_v, fear_v)))
        t_a = stats.zscore(np.concatenate((happy_a, neutral_a, sad_a, fear_a)))
        happy_v = t_v[:happy_v.shape[0]]
        neutral_v = t_v[happy_v.shape[0]:(happy_v.shape[0] + neutral_v.shape[0])]
        sad_v = t_v[(happy_v.shape[0] + neutral_v.shape[0]):(happy_v.shape[0] + neutral_v.shape[0] + sad_v.shape[0])]
        fear_v = t_v[(happy_v.shape[0] + neutral_v.shape[0] + sad_v.shape[0]):]
        happy_a = t_a[:happy_a.shape[0]]
        neutral_a = t_a[happy_a.shape[0]:(happy_a.shape[0] + neutral_a.shape[0])]
        sad_a = t_a[(happy_a.shape[0] + neutral_a.shape[0]):(happy_a.shape[0] + neutral_a.shape[0] + sad_a.shape[0])]
        fear_a = t_a[(happy_a.shape[0] + neutral_a.shape[0] + sad_a.shape[0]):]

        happy_valence.append(happy_v)
        happy_arousal.append(happy_a)
        neutral_valence.append(neutral_v)
        neutral_arousal.append(neutral_a)
        sad_valence.append(sad_v)
        sad_arousal.append(sad_a)
        fear_valence.append(fear_v)
        fear_arousal.append(fear_a)

    plt.scatter(happy_valence,happy_arousal,c='orange',label='happy',alpha=0.5)
    plt.scatter(neutral_valence,neutral_arousal,c='green',label='neutral',alpha=0.5)
    plt.scatter(sad_valence,sad_arousal,c='blue',label='sad',alpha=0.5)
    plt.scatter(fear_valence,fear_arousal,c='red',label='fear',alpha=0.5)
    plt.legend()
    plt.ylabel('Arousal',fontsize=15)
    plt.xlabel('Valence',fontsize=15)
    plt.show()

    return happy_valence,happy_arousal,neutral_valence,neutral_arousal,sad_valence,sad_arousal,fear_valence,fear_arousal