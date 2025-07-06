import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import seaborn as sns
def cca(X, Y, n_splits=4, n_components=2, ran=596):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=ran)

    correlations = []
    p=[]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    x_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(Y)

    x_test_all=[]
    y_test_all=[]

    for train_index, test_index in kf.split(X):
        X_train, X_test = x_scaled[train_index], x_scaled[test_index]
        Y_train, Y_test = y_scaled[train_index], y_scaled[test_index]

        cca = CCA(n_components=n_components)
        cca.fit(X_train, Y_train)

        X_test_c, Y_test_c = cca.transform(X_test, Y_test)
        X_train_c, Y_train_c = cca.transform(X_train, Y_train)

        x_test_all.append(X_test_c)
        y_test_all.append(Y_test_c)


        final_x_test=X_test_c
        final_y_test=Y_test_c
        final_x_train=X_train_c
        final_y_train=Y_train_c
        X_ori_train, X_ori_test = X[train_index], X[test_index]

        corr = pearsonr(X_test_c[:, 0], Y_test_c[:, 0])
        correlations.append(corr[0])

        permuted_corrs = []
        for _ in range(1000):
            permuted_Y = np.random.permutation(Y_test)
            X_test_permuted, Y_test_permuted = cca.transform(X_test, permuted_Y)
            permuted_corr = pearsonr(X_test_permuted[:, 0], Y_test_permuted[:, 0])[0]
            permuted_corrs.append(permuted_corr)
        p_value = np.sum(permuted_corrs > corr[0]) / 1000
        p.append(p_value)

    print(f"\nAverage canonical correlation across folds: {np.mean(correlations):.3f},p-value:{np.mean(p)}")
    return {
        'correlations': correlations,
        'final_x_test': final_x_test,
        'final_y_test': final_y_test,
        'final_x_train': final_x_train,
        'final_y_train': final_y_train,
        'x_test_all': x_test_all,
        'y_test_all': y_test_all,
        'p': p,
        'ori_x_test': X_ori_test,
        'ori_x_train': X_ori_train,
    }


results = cca(x, y, n_components=2)
x_train_c=results['final_x_train']
y_train_c=results['final_y_train']
x_test_c=results['final_x_test']
y_test_c=results['final_y_test']

# plot cross_CCA
fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.regplot(
    x=x_train_c[:,0],
    y=y_train_c[:,0],
    line_kws={"color": "blue", "linewidth": 1.5, "alpha":0.5},
    label='train',
    ax=ax
)
sns.regplot(
    x=x_test_c[:,0],
    y=y_test_c[:,0],
    line_kws={"color": "orange", "linewidth": 1.5, "alpha":0.5},
    label='test',
    ax=ax
)
s,l,r,p,std=stats.linregress(x_train_c[:,0],y_train_c[:,0])
R = r'$r = $' + str(round(np.mean(results['correlations']), 3))
ax.text(0.5, -1.5, R, fontsize=12)
P = r'$p = $'+str(round(np.mean(results['p']), 3))
ax.text(0.5, -1.9, P, fontsize=12)
slope = r'$slope = $'+str(round(s,3))
ax.text(0.5, -1.9, slope, fontsize=12)
ax.set_xlabel('Emotional assessment CCA Variate',fontsize=10)
ax.set_ylabel('Stomach-Brain Coupling CCA Variate',fontsize=10)
ax.legend()
plt.show()
sns.despine()

# CCA loadings
x_ori = np.concatenate((results['ori_x_train'], results['ori_x_test']))
x_c= np.concatenate((results['final_x_train'], results['final_x_test']))
a_loading=np.corrcoef(x_ori[:,0], x_c[:,0])[0,1]
v_loading=np.corrcoef(x_ori[:,1], x_c[:,0])[0,1]