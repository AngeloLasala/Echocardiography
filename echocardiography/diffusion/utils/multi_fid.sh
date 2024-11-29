#!/bin/bash
for w in 0.0 0.2 0.4 0.8; do
    python -m echocardiography.diffusion.evaluation.fid\
    --par_dir "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/diffusion/eco/"\
    --trial trial_2\
    --experiment cond_ldm_6\
    --guide_w $w
    done
