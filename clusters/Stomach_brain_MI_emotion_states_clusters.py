import hparams as hp
import os.path as op
import numpy as np
import pickle
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
import mne
mne.viz.set_browser_backend('qt')
print(__doc__)
from emo_states_clusters import emo_states_clusters
from emo_anova_clusters import emo_anova_clusters


data_path = hp.data_dir
subjects_dir=hp.subjects_mri_dir
src_fname = op.join(subjects_dir , "fsaverage" , "bem" , "fsaverage-ico-5-src.fif")
src_to = mne.read_source_spaces(src_fname)
subjects_pro_meg_dir = hp.subjects_pro_meg_dir
sublist = hp.sublist

num_fre=5
tmin = 0
tstep= 0.0025
tmax = tmin+tstep*(num_fre-1)  # Use a lower tmax to reduce multiple comparisons
num_voxel=2052
n_subjects=16

fsave_vertices = [s["vertno"] for s in src_to]
X = np.zeros((len(fsave_vertices[0])*2, int((tmax-tmin)/tstep+1), n_subjects*3,4))
hap=[]
neu=[]
sad=[]
fea=[]
# Setup for reading the raw data
emotion = True
emo_type=['happy','neutral','sad','fear']
for n in range (n_subjects):

    subname=sublist[n]
    subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
    meegfiletag = '-meg-'
    subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                        + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                        + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                        + str(hp.loose))
    stc_name=op.join(subjects_stc_dir_sub_this,'raw_2017_1')
    stc=mne.read_source_estimate(stc_name)
    stc=stc.crop(tmin=tmin,tmax=tmax)
    morph = mne.compute_source_morph(
        stc,
        subject_from=subname,
        subject_to="fsaverage",
        src_to=src_to,
        subjects_dir=subjects_dir,
    )

    sur_fname = op.join(subjects_stc_dir_sub_this, 'emotion_newMI_fs400_20_5bands_time_suborder.pickle')
    for num_emo in range(4):
        with open(sur_fname, 'rb') as f:
            sur0 = pickle.load(f)[emo_type[num_emo]]
            for num in range(3):
                temp = np.array(sur0[num])[:, :, -1]
                sur_stc1 = stc.copy()
                sur_stc1 = sur_stc1.crop(tmin=tmin, tmax=tmin + tstep * (num_fre - 1))
                sur_stc1.data = np.transpose(temp)
                X[:, :, n * 3 + num, num_emo] = morph.apply(sur_stc1).data

Y = np.transpose(X, [2, 1, 0, 3])
Y = [np.squeeze(x) for x in np.split(Y, 4, axis=-1)]

emo_states_clusters(Y,emo1=2,emo2=1,src_to=src_to,subjects_dir=subjects_dir)
emo_anova_clusters(Y,src_to=src_to,subjects_dir=subjects_dir)