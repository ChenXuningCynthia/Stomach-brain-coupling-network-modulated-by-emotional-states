import hparams as hp
import os.path as op
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

video_file='video_sequence.xlsx'
video_sequence_all=pd.read_excel(video_file)
train_label_all=[]
test_label_all=[]
sublist = hp.sublist
subnum=hp.subnum
trialnum=hp.trialnum
for sb in range(subnum):
    video_sequence = np.floor((np.array(video_sequence_all)[sb]-1)/3)+1
    train_label_all.append(video_sequence)
    subname = sublist[sb]
    for run in range(trialnum):
        result_name = op.join(hp.project_path, 'result', subname, str(run) + '.mat')
        result = io.loadmat(result_name)['result']
        test_label_all.append(result[0, 0])
        test_label_all.append(result[1, 0])

train_label_all=np.array(train_label_all).flatten()
test_label_all=np.array(test_label_all)
cm = confusion_matrix(train_label_all, test_label_all,normalize='true')

# plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='.2g', cmap='Oranges',
            xticklabels=['happy', 'neutral', 'sad', 'fear'],
            yticklabels=['happy', 'neutral', 'sad', 'fear'])
plt.xlabel('Participant Emotion Label',fontsize=10)
plt.ylabel('Stimulus Emotion Label',fontsize=10)
plt.show()