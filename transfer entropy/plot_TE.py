import hparams as hp
import os.path as op
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

subnum = hp.subnum
trialnum=hp.trialnum
sublist = hp.sublist
subjects_pro_meg_dir = hp.subjects_pro_meg_dir
egg_fs=20
meg_fs=400
meg_power_fs=400
num_fs=5

happy_all = []
neutral_all = []
sad_all = []
fear_all = []

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
    mat_name = op.join(subjects_stc_dir_sub_this, 'te_3modes.mat')
    te=io.loadmat(mat_name)
    happy = te['happy']
    neutral = te['neutral']
    sad=te['sad']
    fear=te['fear']

    happy_all.append(happy)
    neutral_all.append(neutral)
    sad_all.append(sad)
    fear_all.append(fear)

happy_all[happy_all<0]=0
neutral_all[neutral_all<0]=0
sad_all[sad_all<0]=0
fear_all[fear_all<0]=0
emo_all=np.concatenate((happy_all,neutral_all,sad_all,fear_all),axis=1)

# Direction S2B and B2S
bands = ['delta','theta','alpha','beta','gamma']
fig, ax = plt.subplots(figsize = (9, 9))

emo_DMN=emo_all[0,:,DMN_parc,2,:].mean(axis=0)
emo_SMN=emo_all[0,:,SMN_parc,2,:].mean(axis=0)
emo_VN=emo_all[0,:,VN_parc,2,:].mean(axis=0)

roi=['DMN','SMN','VN']
custom_palette={'Stomach to Brain':'#FA7F6F','Brain to Stomach':'black'}
data_dict={}
data_dict['DMN']=emo_DMN[:,[1,0]]
data_dict['SMN']=emo_SMN[:,[1,0]]
data_dict['VN']=emo_VN[:,[1,0]]
df_list = []
for roi, arr in data_dict.items():
    temp_df = pd.DataFrame(arr, columns=['Stomach to Brain','Brain to Stomach'])
    temp_df['ROI'] = roi
    temp_df = temp_df.melt(id_vars='ROI', var_name='Direction', value_name='Tranfer Entropy')
    df_list.append(temp_df)
final_df = pd.concat(df_list).reset_index(drop=True)
sns.boxplot(x='ROI',y='Tranfer Entropy',hue='Direction',data=final_df,palette=custom_palette,legend=False)
sns.despine()

# TE frequency distribution
emo_fre=emo_all[:,:,:,2,1].mean(axis=2).mean(axis=1)
roi=['delta','theta','alpha','beta','gamma']
data_dict={}
data_dict['delta']=emo_fre[0,:]
data_dict['theta']=emo_fre[1,:]
data_dict['alpha']=emo_fre[2,:]
data_dict['beta']=emo_fre[3,:]
data_dict['gamma']=emo_fre[4,:]
df_list = []
for roi, arr in data_dict.items():
    temp_df = pd.DataFrame(arr)
    temp_df['ROI'] = roi
    temp_df = temp_df.melt(id_vars='ROI', value_name='Tranfer Entropy')
    df_list.append(temp_df)
final_df = pd.concat(df_list).reset_index(drop=True)
ax=sns.violinplot(x='ROI',y='Tranfer Entropy',data=final_df,palette='Set3')
ax.tick_params(labelsize=20)
ax.set_xlabel('ROI',fontsize=20)
ax.set_ylabel('Tranfer Entropy',fontsize=20)
sns.despine()