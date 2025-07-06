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
from emo_rest_clusters import emo_rest_clusters
from emo_MI import emo_MI

data_path = hp.data_dir
subjects_dir=hp.subjects_mri_dir
src_fname = op.join(subjects_dir , "fsaverage" , "bem" , "fsaverage-ico-5-src.fif")
src_to = mne.read_source_spaces(src_fname)
subjects_pro_meg_dir = hp.subjects_pro_meg_dir
sublist = hp.sublist

num_fre=5
tmin = 0
tstep= 0.0025
tmax = tmin+tstep*(num_fre-1)
num_voxel=2052
n_subjects=16

fsave_vertices = [s["vertno"] for s in src_to]
X = np.zeros((len(fsave_vertices[0])*2, int((tmax-tmin)/tstep+1), n_subjects,4))
hap=[]
neu=[]
sad=[]
fea=[]

hap_rest=[]
neu_rest=[]
sad_rest=[]
fea_rest=[]
# Setup for reading the raw data
emotion = True
emo_type=['happy','neutral','sad','fear']

for n in range (0,n_subjects):
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
    rest_fname = op.join(subjects_stc_dir_sub_this, 'rest_newMI_fs400_20_5bands_time.pickle')
    with open(rest_fname, 'rb') as f:
        rest0 = np.array(pickle.load(f)['mi_all'][1])[:,:,-1]
    for num_emo in range(4):
        with open(sur_fname, 'rb') as f:
            sur0 = pickle.load(f)[emo_type[num_emo]]
        for num in range(len(sur0)):
            temp = np.array(sur0[num])[:, :, -1]
            sur_stc1 = stc.copy()
            sur_stc1 = sur_stc1.crop(tmin=tmin, tmax=tmin + tstep * (num_fre - 1))
            sur_stc1.data = np.transpose(temp)
            sur_stc0=sur_stc1.copy()
            sur_stc0.data = np.transpose(rest0)
            if num_emo==0:
                hap.append(morph.apply(sur_stc1).data)
                hap_rest.append(morph.apply(sur_stc0).data)
            if num_emo==1:
                neu.append(morph.apply(sur_stc1).data)
                neu_rest.append(morph.apply(sur_stc0).data)
            if num_emo == 2:
                sad.append(morph.apply(sur_stc1).data)
                sad_rest.append(morph.apply(sur_stc0).data)
            if num_emo == 3:
                fea.append(morph.apply(sur_stc1).data)
                fea_rest.append(morph.apply(sur_stc0).data)

Y = np.transpose(X, [2, 1, 0, 3])
Y = [np.squeeze(x) for x in np.split(Y, 4, axis=-1)]

emo_rest_clusters(emo=fea,emo_rest=fea_rest,src_to=src_to,subjects_dir=subjects_dir)
emo_MI(stc=stc,emoMI=fea,subname=subname,src_to=src_to,subjects_dir=subjects_dir)