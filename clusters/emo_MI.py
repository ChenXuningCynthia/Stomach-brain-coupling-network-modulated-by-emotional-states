import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
import mne
# emo MI

def emo_MI(stc,emoMI,subname,src_to,subjects_dir):
    morph = mne.compute_source_morph(
            stc,
            subject_from=subname,
            subject_to="fsaverage",
            src_to=src_to,
            subjects_dir=subjects_dir,
        )
    a=morph.apply(stc)
    a.data=np.array(emoMI).mean(axis=0)

    stc_data = a.data
    q95_v = np.quantile(stc_data, q=0.99)
    clim = dict(kind="value", pos_lims=[0, q95_v/2, q95_v])
    brain = a.plot(
        hemi="split",
        views="lateral",
        subjects_dir=subjects_dir,
        time_label="temporal extent (ms)",
        size=(800, 800),
        smoothing_steps=5,
        background='white',
        clim=clim,
    )