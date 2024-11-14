## multi evaluiation of the regression model
for i in {1..9}; do
    for m in ellipses max_value; do
        python test.py --data_path '/media/angelo/OS/Users/lasal/Desktop/DATA_h'\
                        --model_path "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/regression/"\
                        --batch Batch2\
                        --trial trial_$i\
                        --split val\
                        --method_center $m
    done
done
