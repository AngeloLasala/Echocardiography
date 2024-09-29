#!/bin/bash
for w in -1.0 0.0 0.2 0.4 0.6 0.8 1.0 2.0 ; do
    python -m echocardiography.diffusion.evaluation.fid\
    --par_dir "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/diffusion/eco/"\
    --trial trial_2\
    --experiment cond_ldm_1\
    --guide_w $w
    done
