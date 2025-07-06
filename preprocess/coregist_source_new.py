import hparams as hp
import os.path as op
import os
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator,apply_inverse_raw
from mne.coreg import Coregistration
from mne.io import read_info

def coregist_MRIMEG(hp, overwrite=True, plot=False):
    subjects_pro_meg_dir = hp.subjects_pro_meg_dir
    sublist = hp.sublist
    for sb in range(subnum):
        for run in range (trialnum):
            subname = sublist[sb]
            subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
            filetocheck = op.join(subjects_pro_meg_dir_sub, 'all' + hp.raw_MEG_name + str(run) + 'new-epo-trans.fif')
            if op.exists(filetocheck) and (not overwrite):
                continue
            else:
                megfile = op.join(subjects_pro_meg_dir_sub, 'run'+ str(run) +'_meg_tsss.fif')
                _coregist_MRIMEG(hp, subname, megfile, filetocheck, plot,run)

def _coregist_MRIMEG(hp, subname, megfile, filetocheck, plot,run):
    info = read_info(megfile)
    fiducials = "estimated"  # get fiducials from fsaverage
    coreg = Coregistration(info, subject=subname, subjects_dir=hp.subjects_mri_dir, fiducials=fiducials)
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2., verbose=True, eeg_weight=0.)
    coreg.omit_head_shape_points(distance=5. / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True, eeg_weight=0.)
    mne.write_trans(filetocheck, coreg.trans,overwrite=True)
    if plot:
        plot_kwargs = dict(subject=subname, subjects_dir=hp.subjects_mri_dir,
                           surfaces="head-dense", dig=True, eeg='original',
                           meg='sensors', show_axes=True,
                           coord_frame='meg')
        view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                           focalpoint=(0., 0., 0.))
        fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
    print('test')

def get_forwardmodel(hp, overwrite):
    subjects_pro_meg_dir = hp.subjects_pro_meg_dir
    sublist = hp.sublist
    for sb in range(subnum):
        for run in range(trialnum):
            subname = sublist[sb]
            subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
            filetocheck = op.join(subjects_pro_meg_dir_sub, 'all' + hp.raw_MEG_name + str(run) + 'new-epo-meg-' \
                                  + hp.spacing + '-ico-' + str(hp.ico_downsampling) + '-fwd.fif')
            isexist = op.exists(filetocheck) and (not overwrite)
            if not isexist:
                _get_forwardmodel(hp, subname,run)

def _get_forwardmodel(hp, subname,run):
    subjects_pro_meg_dir_sub = op.join(hp.subjects_pro_meg_dir, subname)
    src = mne.setup_source_space(subject=subname, spacing=hp.spacing, add_dist='patch',
                                 subjects_dir=hp.subjects_mri_dir)

    conductivity = (0.3,)
    model = mne.make_bem_model(subject=subname, ico=hp.ico_downsampling,
                               conductivity=conductivity,
                               subjects_dir=hp.subjects_mri_dir)
    bem = mne.make_bem_solution(model)
    megfile = op.join(subjects_pro_meg_dir_sub, 'run'+ str(run) +'_meg_tsss.fif')
    transfile = op.join(subjects_pro_meg_dir_sub, 'all' + hp.raw_MEG_name + str(run)+'new-epo-trans.fif')
    fwdfile = op.join(subjects_pro_meg_dir_sub, 'all' + hp.raw_MEG_name + str(run)+'new-epo-meg-' \
                      + hp.spacing + '-ico-' + str(hp.ico_downsampling) + '-fwd.fif')
    srcfile = op.join(subjects_pro_meg_dir_sub, 'all' + hp.raw_MEG_name +str(run)+ 'new-epo-meg-' \
                      + hp.spacing + '-ico-' + str(hp.ico_downsampling) + '-src.fif')
    # if distance of sources of each voxel less than mindist, it will be Delete.
    fwd = mne.make_forward_solution(megfile, trans=transfile, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=0.0, n_jobs=1,
                                    verbose=True)
    mne.write_forward_solution(fname=fwdfile, fwd=fwd, overwrite=True)
    src.save(fname=srcfile,overwrite=True)

def estimate_source(hp, avgepoch=False, overwrite=True):
    sublist = hp.sublist
    for sb in range(subnum):
        for run in range(trialnum):
            subname = sublist[sb]
            subjects_stc_dir_sub = op.join(hp.subjects_stc_dir, subname)
            meegfiletag = '-meg-'
            subjects_stc_dir_sub_this = op.join(subjects_stc_dir_sub, hp.processed_MEG_name, hp.stc_method \
                                                + '-' + hp.covmethod + '-snr' + str(hp.snr) + meegfiletag \
                                                + hp.spacing + 'new-ico' + str(hp.ico_downsampling) + '-losse' \
                                                + str(hp.loose))
            if overwrite:
                isexist = False
            else:
                isexist = True
                if not op.exists(op.join(subjects_stc_dir_sub_this, '-a-lh.stc')):
                    isexist = False
                    break
            if isexist:
                continue
            else:
                _estimate_source(hp, subname, meegfiletag,run,subjects_stc_dir_sub_this=subjects_stc_dir_sub_this)


def _estimate_source(hp, subname, meegfiletag,  run,subjects_stc_dir_sub_this):
    subjects_pro_meg_dir_sub = op.join(hp.subjects_pro_meg_dir, subname)
    subjects_raw_meg_dir_sub = op.join(hp.data_dir, subname)

    if not op.exists(subjects_stc_dir_sub_this):
        os.makedirs(subjects_stc_dir_sub_this)

    megfile = op.join(subjects_pro_meg_dir_sub, 'run'+str(run)+'_meg_tsss.fif')
    megfile_stim = op.join(subjects_raw_meg_dir_sub, 'run' + str(run) + '_tsss.fif')
    basefile = op.join(subjects_pro_meg_dir_sub, 'run1_meg_tsss.fif')
    fwdfile = op.join(subjects_pro_meg_dir_sub, 'all' + hp.raw_MEG_name+ str(run)  + 'new-epo' + meegfiletag \
                      + hp.spacing + '-ico-' + str(hp.ico_downsampling) + '-fwd.fif')

    raw = mne.io.read_raw_fif(megfile, allow_maxshield=False).load_data()
    raw_stim = mne.io.read_raw_fif(megfile_stim, allow_maxshield=False).load_data()
    raw_base = mne.io.read_raw_fif(basefile, allow_maxshield=False).load_data()

    events = mne.find_events(raw_stim)
    events=np.reshape(events[0,:],(-1,3))
    events[0, 0] = int(events[0, 0]/1000*400)
    event_id=dict(evoke=1)

    reject=dict(grad=4000e-13, mag=400e-12)
    epochs_all = mne.Epochs(raw, events, event_id, tmin=0, tmax=raw.times[-1]-(events[0,0]/400-raw.first_time), baseline=(0,5),reject=reject)
    data=epochs_all.get_data()

    raw_base = raw_base.copy().crop(tmin=200, tmax=None, include_tmax=True)
    fwd = mne.read_forward_solution(fwdfile)
    covfile = op.join(subjects_pro_meg_dir_sub, 'all_' + meegfiletag + subname+str(run)+ 'base_new_cov.fif')

    if op.exists(covfile):
        cov = mne.read_cov(covfile)
    else:
        # tstep need change in each run
        cov = mne.compute_raw_covariance(raw_base, tmin=0, tmax=None, method=['shrunk', 'empirical'])
        mne.write_cov(covfile, cov)

    inverse_operator = make_inverse_operator(epochs_all.info, fwd, cov, loose=hp.loose, depth=0.8, rank='info')

    #反向计算
    stc= apply_inverse_raw(epochs_all._raw, inverse_operator, hp.lambda2,method=hp.stc_method, use_cps=True)
    rawname='stc_'+str(run)
    stc.save(fname=op.join(subjects_stc_dir_sub_this, rawname ), ftype='stc',overwrite=True)


if __name__ == '__main__':
    subnum = hp.subnum
    trialnum = hp.trialnum
    coregist_MRIMEG(hp, overwrite=True, plot=False)
    get_forwardmodel(hp, overwrite=True)
    estimate_source(hp, overwrite=True)










