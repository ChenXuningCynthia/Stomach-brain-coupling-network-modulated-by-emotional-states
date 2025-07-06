import hparams as hp
import os.path as op
import mne
import pickle
import numpy as np

numsub=hp.subnum
subjects_dir=hp.subjects_mri_dir
src_fname = op.join(subjects_dir , "fsaverage" , "bem" , "fsaverage-ico-5-src.fif")
src_to = mne.read_source_spaces(src_fname)
sublist = hp.sublist
subjects_pro_meg_dir = hp.subjects_pro_meg_dir

egg_fs=20
meg_fs=400
meg_power_fs=200
num_fs=100
num_voxel = 2052
num_time=12
tmin = 0
tstep= 0.0025
tmax = tmin+tstep*(num_time-1)
happy_all = []
neutral_all = []
sad_all = []
fear_all = []

def SBMI_distribution():
    for sb in range(numsub):
        subname = sublist[sb]
        print(sb, subname)
        subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
        subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
        meegfiletag = '-meg-'
        subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                            + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                            + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                            + str(hp.loose))
        stc_name = op.join(subjects_stc_dir_sub_this, 'raw_2017_1')
        stc = mne.read_source_estimate(stc_name)
        stc = stc.crop(tmin=tmin, tmax=tmax)
        sur_fname=op.join(subjects_stc_dir_sub_this, 'emotion_newMI_fs400_20_5bands_time_suborder.pickle')

        with open(sur_fname, 'rb') as f:
            happy = pickle.load(f)['happy']
        with open(sur_fname, 'rb') as f:
            neutral = pickle.load(f)['neutral']
        with open(sur_fname, 'rb') as f:
            sad = pickle.load(f)['sad']
        with open(sur_fname, 'rb') as f:
            fear = pickle.load(f)['fear']

        happy_all.append(happy[:,:, -1])
        neutral_all.append(neutral[:,:, -1])
        sad_all.append(sad[:,:, -1])
        fear_all.append(fear[:,:, -1])

    sur_all=np.concatenate((happy_all,neutral_all,sad_all,fear_all),axis=0)
    T,clusters,cluster_p_values,H0=pickle.load(open('negative_neutral_cluster/clu_newMI_last1min_anova_5bands.pickle', 'rb'))['clu']
    good_clusters = np.where(cluster_p_values < 0.05)[0]
    map=np.zeros([5,20484])
    for i in range(len(good_clusters)):
        clu0=clusters[good_clusters[i]]
        for j in range(len(clu0[0])):
            map[clu0[0][j],clu0[1][j]]=T[clu0[0][j],clu0[1][j]]

    sur_all_mean=[]
    for i in range(sur_all.shape[0]):
        sur_fre=np.zeros(5)
        temp_sur=sur_all[i,:,:]
        temp_sur[map==0]=0
        for j in range(5):
            sur_fre[j]=np.mean(temp_sur[j,:][temp_sur[j,:]!=0])
        sur_all_mean.append(sur_fre)
    sur_all_mean=np.array(sur_all_mean)
    return sur_all_mean



