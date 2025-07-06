import os.path as op
import os
import mne
import hparams as hp
from egg_preprocess import egg_preprocess
from meg_preprocess import meg_preprocess

def process(hp, overwrite=False):
    subjects_pro_meg_dir = hp.subjects_pro_meg_dir
    if not op.exists(subjects_pro_meg_dir):
        os.makedirs(subjects_pro_meg_dir)
    sublist = hp.sublist
    numsub=hp.subnum
    numrun=hp.numrun
    for sb in range(numsub):
        subname = sublist[sb]
        print(subname)
        subjects_pro_meg_dir_sub = op.join(subjects_pro_meg_dir, subname)
        if not op.exists(subjects_pro_meg_dir_sub):
            os.mkdir(subjects_pro_meg_dir_sub)
        for run in range(numrun):
            megfile = op.join(subjects_pro_meg_dir_sub, 'run' + str(run) +  hp.processed_MEG_name + '.fif')
            eggfile = op.join(subjects_pro_meg_dir_sub, 'run' + str(run) +  hp.processed_EGG_name + '.fif')
            if op.exists(megfile) and op.exists(eggfile) and(not overwrite):
                continue
            else:
                subjects_raw_meg_dir_sub = op.join(hp.data_dir, subname)
                rawdir = op.join(subjects_raw_meg_dir_sub, 'run' + str(run) + hp.raw_MEG_name + '.fif')
                rawfile = mne.io.read_raw_fif(rawdir, preload=True)
                egg_preprocess(rawfile, eggfile)
                meg_preprocess(rawfile,megfile)

if __name__ == '__main__':
    process(hp=hp,overwrite=True)