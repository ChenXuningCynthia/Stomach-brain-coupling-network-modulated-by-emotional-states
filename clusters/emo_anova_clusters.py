import numpy as np
from mne.stats import (
    f_mway_rm,
    f_threshold_mway_rm,
    spatio_temporal_cluster_test,
    summarize_clusters_stc,
)
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
import mne
mne.viz.set_browser_backend('qt')
print(__doc__)

def stat_fun(*args):
    # get f-values only.
    return f_mway_rm(
        np.swapaxes(args, 1, 0),
        factor_levels=[4],
        effects="A",
        return_pvals=False,
    )[0]
def emo_anova_clusters(Y,src_to,subjects_dir):
    n_subjects=16
    print("Computing adjacency.")
    adjacency = mne.spatial_src_adjacency(src_to)
    pthresh = 0.01
    f_thresh = f_threshold_mway_rm(n_subjects*3, [4], "A", pthresh)
    n_permutations = 1000

    print("Clustering.")
    F_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_test(
        Y,
        adjacency=adjacency,
        n_jobs=None,
        threshold=f_thresh,
        stat_fun=stat_fun,
        n_permutations=n_permutations,
        buffer_size=None,
        seed=None,
        max_step=5,
    )
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    print("Visualizing clusters.")
    map = np.zeros([5, 20484])
    for j in range(len(good_cluster_inds)):
        clu0=clusters[good_cluster_inds[j]]
        for i in range(len(clu0[0])):
            map[clu0[0][i], clu0[1][i]] = F_obs[clu0[0][i], clu0[1][i]]
    fsave_vertices = [s["vertno"] for s in src_to]
    stc_all_cluster_vis = summarize_clusters_stc(
        clu, p_thresh=0.05,tstep=0.0025, vertices=fsave_vertices, subject="fsaverage"
    )
    stc_mean=stc_all_cluster_vis.copy().crop(tmin=0,tmax=0.01)
    stc_mean.data=np.transpose(map)
    stc_mean.save(fname='anova', ftype='stc',overwrite=True)

    stc_data = stc_mean.data
    q95_v = np.quantile(stc_data, q=0.99)
    clim = dict(kind="value", pos_lims=[0, q95_v/2, q95_v])
    brain = stc_mean.plot(
        hemi="split",
        views="lateral",
        subjects_dir=subjects_dir,
        time_label="temporal extent (ms)",
        size=(800, 800),
        smoothing_steps=5,
        background='white',
        clim=clim,
    )