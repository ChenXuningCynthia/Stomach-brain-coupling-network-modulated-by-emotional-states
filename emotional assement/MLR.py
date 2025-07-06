import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns

def MLR(x,y,happy_arousal,happy_valence,h_y,neutral_arousal,neutral_valence,n_y,sad_arousal,sad_valence,s_y,fear_arousal,fear_valence,f_y):
    model=LinearRegression()
    model.fit(x,y)
    k=model.coef_
    s,l,r,p,std=stats.linregress(k[0]*x[0]+k[1]*x[1],y)
    fig, ax = plt.subplots(figsize = (4, 3))
    plt.scatter(k[0]*happy_arousal+k[1]*happy_valence,h_y,c='orange',label='happy',alpha=0.5)
    plt.scatter(k[0]*neutral_arousal+k[1]*neutral_valence,n_y,c='green',label='neutral',alpha=0.5)
    plt.scatter(k[0]*sad_arousal+k[1]*sad_valence,s_y,c='blue',label='sad',alpha=0.5)
    plt.scatter(k[0]*fear_arousal+k[1]*fear_valence,f_y,c='red',label='fear',alpha=0.5)

    x_s=k[0]*x[0]+k[1]*x[1]
    plt.plot(x_s,s*x_s+l,color='black')
    L = r'$ x= $' +str(np.format_float_scientific(k[0],precision=1))+r'$ aro $' +str(np.format_float_scientific(k[1],precision=1))+r'$ val$'

    R = r'$r = $'+str(round(r,4))
    plt.text(-0.0015, 0.035, R, fontsize=10)
    P = r'$p = $'+str(round(p,4))
    plt.text(-0.0015, 0.032, P, fontsize=10)
    S = r'$slope = $'+str(round(s,4))
    ax=plt.gca()
    ax.ticklabel_format(style='sci',scilimits=(-1,2),axis='x', useMathText=True)
    ax.ticklabel_format(style='sci',scilimits=(-1,2),axis='y', useMathText=True)
    plt.legend()
    plt.xlabel('Valence & Arousal',fontsize=10)
    plt.ylabel('Modulation Index',fontsize=10)
    plt.show()
    sns.despine()