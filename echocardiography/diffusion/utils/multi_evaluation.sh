#!/bin/bash

for w in 0.8 1.0; do
    for epoch in 20 40 60 100 120 150; do
        python -m echocardiography.diffusion.evaluation.hypertropy_eval\
                --par_dir "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/diffusion/eco"\
                --par_dir_regression "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/regression"\
                --trial_regression trial_3\
                --trial trial_2\
                --experiment cond_ldm_8\
                --guide_w $w\
                --epoch $epoch
    done
done