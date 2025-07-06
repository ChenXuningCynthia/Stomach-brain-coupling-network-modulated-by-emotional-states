from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test
import numpy as np
from mne.stats import summarize_clusters_stc
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
import mne
mne.viz.set_browser_backend('qt')

def emo_states_clusters(Y,emo1,emo2,src_to,subjects_dir):
    n_subjects=16
    emo_diff=Y[emo1]-Y[emo2]
    adjacency = mne.spatial_src_adjacency(src_to)
    p_threshold = 0.01
    df = n_subjects*3 - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    T_obs, clusters, cluster_p_values, H0 = clu_sn = spatio_temporal_cluster_1samp_test(
        emo_diff,
        adjacency=adjacency,
        n_jobs=None,
        threshold=t_threshold,
        n_permutations=1000,
        buffer_size=None,
        verbose=True,
    )
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    map = np.zeros([5, 20484])
    for j in range(len(good_cluster_inds)):
        clu0=clusters[good_cluster_inds[j]]
        for i in range(len(clu0[0])):
            map[clu0[0][i], clu0[1][i]] = T_obs[clu0[0][i], clu0[1][i]]
    fsave_vertices = [s["vertno"] for s in src_to]
    stc_all_cluster_vis = summarize_clusters_stc(
        clu_sn, p_thresh=0.05,tstep=0.0025, vertices=fsave_vertices, subject="fsaverage"
    )

    stc_mean=stc_all_cluster_vis.copy()
    stc_mean.data=np.transpose(map)
    stc_data = stc_mean.data
    q95_v = np.quantile(stc_data, q=0.99)
    clim = dict(kind="value", pos_lims=[0, q95_v / 2, q95_v])
    brain = stc_all_cluster_vis.plot(
        hemi="split",
        views="lateral",
        # fsaverage_dir?
        subjects_dir=subjects_dir,
        time_label="temporal extent (ms)",
        size=(800, 800),
        smoothing_steps=5,
        background='white',
        clim=clim,
    )