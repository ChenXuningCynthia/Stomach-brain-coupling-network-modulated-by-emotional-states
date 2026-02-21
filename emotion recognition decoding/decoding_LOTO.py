import numpy as np
import hparams as hp
import os.path as op
import scipy.io as io
import pickle
import random
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore
from model import svm_classification,knn_classification

def dataloader_trial(subname,fre,result_all,test_id):
    subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
    meegfiletag = '-meg-'
    subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                        + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                        + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                        + str(hp.loose))
    sur_fname = op.join(subjects_stc_dir_sub_this, 'emotion_newMI_fs400_20_5bands_time.pickle')
    feature = []
    label = []
    test_feature = []
    test_label = []
    with open(sur_fname, 'rb') as f:
        clu = pickle.load(f)['mi_all']
    for num_emo in range(12):
        va_label =result_all[num_emo]
        if va_label==4:
            continue
        emo_label=int(va_label/4)
        if fre=='all':
            emo_data = (np.array(clu[num_emo]))
            emo_data = emo_data.reshape(-1,emo_data.shape[-1])
        else:
            emo_data = np.array(clu[num_emo])[fre, :, :]

        if num_emo == test_id:
            test_feature.extend(np.transpose(emo_data))
            test_label.extend(np.ones(emo_data.shape[1]) * emo_label)
        else:
            feature.extend(np.transpose(emo_data))
            label.extend(np.ones(emo_data.shape[1]) * emo_label)
    if all(x==label[0] for x in label):
        return True,True,True,True
    else:
        all_feature = feature.copy()
        all_feature.extend(test_feature)
        all_feature = zscore(all_feature)
        train_feature=all_feature[:len(feature),:]
        test_feature_z=all_feature[-len(test_feature):,:]
        for tmp in range(2):
            num_t = label.count(tmp)
            if num_t<6:
                ind=np.where(label==np.float64(tmp))[0]
                train_feature.extend(train_feature[i] for i in ind)
                label.extend(np.ones(num_t) * tmp)
                if num_t<3:
                    train_feature.extend(train_feature[i] for i in ind)
                    label.extend(np.ones(num_t) * tmp)

        smote = SMOTE(random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(train_feature, label)
        return list(X_oversampled),y_oversampled,list(test_feature_z),test_label

def leave_one_trail_out():
    sub_acc=0
    fre='all'
    sublist = hp.sublist
    num_sub=16
    for sb in range(0,num_sub):
        accuracy = 0
        subname = sublist[sb]
        print(sb, subname)
        result_all = np.zeros([12])
        for run in range(2, 8):
            result_name = op.join(hp.project_path,'result', subname, str(run) + '.mat')
            result = io.loadmat(result_name)['result']
            result_all[(run - 2)*2] = result[0, 1]
            result_all[(run - 2)*2+ 1] = result[1, 1]
        num_tr=0
        for trial in range(12):
            if result_all[trial]==4:
                continue
            train_feature,train_label,test_feature,test_label=dataloader_trial(subname,fre,result_all,trial)
            if train_feature==True:
                continue
            else:
                num_tr += 1
            # 定义分类器
            index = [i for i in range(len(train_label))]
            random.shuffle(index)
            train_feature=np.array(train_feature)
            train_label =np.array(train_label)
            train_feature = train_feature[index,:]
            train_label = train_label[index]

            cur_accuracy = svm_classification(train_feature,train_label,test_feature,test_label)
            cur_accuracy=knn_classification(train_feature,train_label,test_feature,test_label)
            accuracy += cur_accuracy['acc']
            print('当前 experiment 的 accuracy 为：{}'.format(cur_accuracy['acc']))
        print('当前 sub 的 accuracy 为：{}'.format(accuracy / num_tr))
        sub_acc+=accuracy/num_tr
    print('所有 experiment 上的平均 accuracy 为：{}'.format(sub_acc / num_sub))

if __name__ == '__main__':
    leave_one_trail_out()
