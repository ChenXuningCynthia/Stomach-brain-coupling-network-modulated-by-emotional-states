from sklearn import svm
import hparams as hp
import os.path as op
import scipy.io as io
import random
import sys
from scipy.stats import zscore
from scipy.io import savemat
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
)
from modle import svm_classification,knn_classification
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import classification_report


def evaluate_performance(y_true, y_pred, y_score, title, save_prefix, sub_acc_array):
    """输出各类指标与图像，添加per-class metrics和chance level比较"""
    chance_level = 0.5
    os.makedirs(save_prefix, exist_ok=True)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        # SVM: decision_function → 已经是正类得分
        pos_class_score = y_score.ravel()
    elif y_score.ndim == 2:
        # KNN: predict_proba → 取正类（label=1）的概率
        if y_score.shape[1] != 2:
            raise ValueError(f"Expected 2 columns in y_score for binary classification, got {y_score.shape[1]}")
        pos_class_score = y_score[:, 1].ravel()  # 取第2列（正类概率）
    acc = np.mean(y_true == y_pred)
    sem = np.std(sub_acc_array) / np.sqrt(len(sub_acc_array))
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    prec_micro = precision_score(y_true, y_pred, average='micro')
    rec_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    auc_val = roc_auc_score(y_true, pos_class_score)

    # Per-class metrics (for binary: class 0 and 1)
    prec_per_class = precision_score(y_true, y_pred, average=None)
    rec_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    print(f"--- {title} ---")
    print(f"Accuracy: {acc:.4f} (Chance Level: {chance_level:.4f})")
    print(f"Macro Avg - Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Micro Avg - Precision: {prec_micro:.4f}, Recall: {rec_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"Per-Class - Precision: {prec_per_class}, Recall: {rec_per_class}, F1: {f1_per_class}")
    print(f"AUC: {auc_val:.4f}")
    print(classification_report(y_true, y_pred))  # Detailed per-class report

    # 混淆矩阵 (使用seaborn for better viz if available)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='.2', cmap='Blues', cbar=True)
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{save_prefix}/confmat.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, pos_class_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label=f'Chance Level (AUC={chance_level:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} ROC')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_prefix}/roc.png', dpi=300, bbox_inches='tight')
    plt.close()

    summary_dict = {
        'Accuracy': acc, 'SEM_Accuracy': sem,
        'Precision_macro': prec_macro, 'Recall_macro': rec_macro, 'F1_macro': f1_macro,
        'Precision_micro': prec_micro, 'Recall_micro': rec_micro, 'F1_micro': f1_micro,
        'Precision_per_class': prec_per_class.tolist(),  # Convert to list for saving
        'Recall_per_class': rec_per_class.tolist(),
        'F1_per_class': f1_per_class.tolist(),
        'AUC': auc_val,
        'Confusion_Matrix': cm.tolist(),
        'All_Accuracies': sub_acc_array,  # For further analysis
        'All_y_true': y_true,
        'All_y_pred': y_pred,
        'All_y_score': y_score
    }

    # Pickle
    with open(os.path.join(save_prefix, 'results.pkl'), 'wb') as f:
        pickle.dump(summary_dict, f)

    return summary_dict


def dataloader_sub(subname,fre,result_all,datatype,datadir,va):
    subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
    meegfiletag = '-meg-'
    subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                        + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                        + hp.spacing + '-ico' + str(hp.ico_downsampling) + '-losse' \
                                        + str(hp.loose))
    num_name = op.join(subjects_stc_dir_sub_this, 'num_emo.mat')
    sur_fname = op.join(subjects_stc_dir_sub_this, datadir)

    feature = []
    label = []

    with open(sur_fname, 'rb') as f:
        clu = pickle.load(f)[datatype]
    for num_emo in range(12):
        av_label =result_all[num_emo]
        if va == 'valence':
            if av_label >= 4:
                emo_label = 1
            else:
                emo_label = 0
        else:
            if av_label > 4:
                emo_label = 1
            else:
                emo_label = 0

        if fre == 'all':
            emo_data = (np.array(clu[num_emo]))[:, :, -3:]
            emo_data = emo_data.reshape(-1, emo_data.shape[-1])
        else:
            emo_data = np.array(clu[num_emo])[fre, :, -3:]
        feature.extend(np.transpose(emo_data))
        label.extend(np.ones(emo_data.shape[1]) * emo_label)
    # feature=zscore(feature)
    return feature,label


def svm_leave_one_sub_out(fre = 0,datatype='meg_de',method='SVM',va='valence',datadir='emotion_MEG_DE_PSD_20Hz_1min_normalorder.pickle'):
    '''
        按照 SEED 数据集原始论文中的 SVM 计算方式测试准确率和方差，每个 experiment 分开计算，取其中 9 个 trial 为训练集，6 个 trial 为测试集
    :param folder_path: ExtractedFeatures 文件夹路径
    :return None:
    '''
    accuracy = 0
    sub_acc = []
    sublist = hp.sublist
    num_sub = 24
    y_pred_all = []
    y_score_all = []
    y_true_all = []
    if va=='valence':
        id=1
    else:
        id=2

    for oneout in range(num_sub):
        train_feature = []
        train_label = []
        test_feature = []
        test_label = []
        outname = sublist[oneout]
        print(oneout, outname)
        for sb in range(24):
            subname = sublist[sb]
            result_all = np.zeros([12])
            for run in range(2, 8):
                result_name = op.join(hp.project_path_dir,'result', subname, str(run) + '.mat')
                # result_name = op.join('result', subname, str(run) + '.mat')
                result = io.loadmat(result_name)['result']
                result_all[(run - 2) * 2] = result[0, id]
                result_all[(run - 2) * 2 + 1] = result[1, id]

            feature,label=dataloader_sub(subname,fre,result_all,datatype,datadir,va)
            pca=PCA(n_components=34)
            feature=pca.fit_transform(feature)
            if sb!=oneout:
                train_feature.extend(feature)
                train_label.extend(label)
            else:
                test_feature.extend(feature)
                test_label.extend(label)
        # 定义 svm 分类器
        smote = SMOTE(random_state=42)
        train_feature_oversampled, train_label_oversampled = smote.fit_resample(train_feature, train_label)
        index = [i for i in range(len(train_label_oversampled))]
        random.shuffle(index)
        train_feature=np.array(train_feature_oversampled)
        train_label =np.array(train_label_oversampled)
        train_feature = train_feature[index,:]
        train_label = train_label[index]
        if method == 'KNN':
            cur_accuracy = knn_classification(train_feature, train_label, test_feature, test_label)
        else:
            cur_accuracy = svm_classification(train_feature, train_label, test_feature, test_label)
        accuracy += cur_accuracy['acc']
        print('当前 experiment 的 accuracy 为：{}'.format(cur_accuracy['acc']))

        y_pred_all.extend(cur_accuracy['p_label'])
        y_score_all.extend(cur_accuracy['decision_val'])
        y_true_all.extend(test_label)

        sub_acc.append(cur_accuracy['acc'])


    sub_acc_array = np.array(sub_acc)
    title = f"{va}_{datatype}_{fre}"
    save_prefix = op.join(hp.project_path_dir,'decoding_results','LOSO',method, title)  # 假设图像目录
    trial_metrics = evaluate_performance(y_true_all, y_pred_all, y_score_all, title, save_prefix, sub_acc_array)
    print(f"All subs mean acc: {np.mean(sub_acc_array):.4f}")
    return trial_metrics


svm_leave_one_sub_out(fre='all', datatype='meg_psd', method='SVM', va='arousal',datadir='emotion_MEG_DE_PSD.pickle')

print('end')
