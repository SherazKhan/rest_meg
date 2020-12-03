import mne
import os.path as op


subjects_dir = r'D:\Dropbox (Personal)\banu\anat'

mne.set_config('SUBJECTS_DIR',subjects_dir)

def process_subject_bem(subject, spacing='ico5'):
    mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True, volume='T1', atlas=True,
                       gcaatlas=False, preflood=None)
    conductivity = (0.3,)
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False)
    bem_fname = op.join(subjects_dir,subject,'bem', '%s-src.fif' % subject)
    src_fname = op.join(subjects_dir, subject, 'bem', '%s-src.fif' % spacing)
    mne.write_bem_solution(bem_fname, bem=bem)
    mne.write_source_spaces(src_fname, src=src)



process_subject_bem(100002)







