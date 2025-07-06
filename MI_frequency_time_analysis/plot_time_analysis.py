import numpy as np
from scipy import stats as stats
import hparams as hp
import os.path as op
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline
data_path = hp.data_dir
subjects_dir=hp.subjects_mri_dir
subjects_pro_meg_dir = hp.subjects_pro_meg_dir
sublist = hp.sublist
numsub = hp.subnum
num_voxel=2052
n_subjects=16
emo_type=['happy','neutral','sad','fear']
video_file='video_sequence.xlsx'
video_sequence_all=pd.read_excel(video_file)

time_MI=[]
happy_all=[]
neutral_all=[]
sad_all=[]
fear_all=[]
short_fear=[]
mi_all=[]
for n in range (numsub):
    time=[]
    video_sequence = np.array(video_sequence_all)[n]-1
    subname=sublist[n]
    subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
    meegfiletag = '-meg-'
    subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                        + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                        + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                        + str(hp.loose))
    sur_fname = op.join(subjects_stc_dir_sub_this, 'emotion_SBMI.pickle')

    with open(sur_fname, 'rb') as f:
        happy = pickle.load(f)['happy']
    with open(sur_fname, 'rb') as f:
        neutral = pickle.load(f)['neutral']
    with open(sur_fname, 'rb') as f:
        sad = pickle.load(f)['sad']
    with open(sur_fname, 'rb') as f:
        fear = pickle.load(f)['fear']


    for i in range(len(happy)):
        happy_all.append(np.array(happy[i]).mean(axis=1)[:, -8:])
    for i in range(len(neutral)):
        neutral_all.append(np.array(neutral[i]).mean(axis=1)[:, -8:])
    for i in range(len(sad)):
        sad_all.append(np.array(sad[i]).mean(axis=1)[:,-8:])
    for i in range(len(fear)):
        fear_all.append(np.array(fear[i]).mean(axis=1)[:, -8:])

i=0
happy_time=np.array(happy_all)[:,i,:]
neutral_time=np.array(neutral_all)[:,i,:]
sad_time=np.array(sad_all)[:,i,:]
fear_time=np.array(fear_all)[:,i,:]

h_mean=np.mean(happy_time,axis=0)
h_sem=stats.sem(happy_time,axis=0,ddof=0)
n_mean=np.mean(neutral_time,axis=0)
n_sem=stats.sem(neutral_time,axis=0,ddof=0)
s_mean=np.mean(sad_time,axis=0)
s_sem=stats.sem(sad_time,axis=0,ddof=0)
f_mean=np.mean(fear_time,axis=0)
f_sem=stats.sem(fear_time,axis=0,ddof=0)

time=np.arange(0,71,10)
x_smooth = np.linspace(time.min(), time.max(), 300)
spline = make_interp_spline(time, h_mean)
h_mean_smooth = spline(x_smooth)
spline = make_interp_spline(time, n_mean)
n_mean_smooth = spline(x_smooth)
spline = make_interp_spline(time, s_mean)
s_mean_smooth = spline(x_smooth)
spline = make_interp_spline(time, f_mean)
f_mean_smooth = spline(x_smooth)

spline = make_interp_spline(time, h_sem)
h_sem_smooth = spline(x_smooth)
spline = make_interp_spline(time, n_sem)
n_sem_smooth = spline(x_smooth)
spline = make_interp_spline(time, s_sem)
s_sem_smooth = spline(x_smooth)
spline = make_interp_spline(time, f_sem)
f_sem_smooth = spline(x_smooth)

# plot-emotion-time-analysis
plt.figure(figsize=(4, 2.5))
plt.plot(x_smooth, h_mean_smooth, label='happy', color='orange',lw=3)
plt.fill_between(x_smooth, h_mean_smooth -h_sem_smooth, h_mean_smooth+h_sem_smooth, color='orange', alpha=0.2)

plt.plot(x_smooth, n_mean_smooth, label='neutral', color='green',lw=3)
plt.fill_between(x_smooth, n_mean_smooth -n_sem_smooth, n_mean_smooth+n_sem_smooth, color='green', alpha=0.1)

plt.plot(x_smooth, s_mean_smooth, label='sad', color='blue',lw=3)
plt.fill_between(x_smooth, s_mean_smooth -s_sem_smooth, s_mean_smooth+s_sem_smooth, color='blue', alpha=0.1)

plt.plot(x_smooth, f_mean_smooth, label='fear', color='red',lw=3)
plt.fill_between(x_smooth, f_mean_smooth -f_sem_smooth, f_mean_smooth+f_sem_smooth, color='red', alpha=0.1)

plt.xlabel('Time',fontsize=10)
plt.ylabel('Modulation Index',fontsize=10)
plt.legend(loc='upper left',fontsize=10)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2),useMathText=True)
sns.despine()

# plot-emotion-frequency-analysis
emotions=['happy','neutral','sad','fear']
data_dict={}
data_dict['happy']=np.array(happy_all)
data_dict['neutral']=np.array(neutral_all)
data_dict['sad']=np.array(sad_all)
data_dict['fear']=np.array(fear_all)
df_list = []
for emotion, arr in data_dict.items():
    temp_df = pd.DataFrame(arr, columns=['delta','theta','alpha','beta','gamma'])
    temp_df['Emotions'] = emotion
    temp_df = temp_df.melt(id_vars='Emotions', var_name='frequency band', value_name='Modulation Index')
    df_list.append(temp_df)
final_df = pd.concat(df_list).reset_index(drop=True)
ax=sns.violinplot(x='Emotions',y='Modulation Index',hue='frequency band',data=final_df,palette="Set3",legend=False)
ax.tick_params(labelsize=20)
ax.set_xlabel('Emotions',fontsize=20)
ax.set_ylabel('Modulation Index',fontsize=20)
sns.despine()
