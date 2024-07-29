# Description: Script to train the model for the regression task using the repository as a module
# running all the code from the 'Echocardiography' folder
# general info about the DATA
#
# local folder: --data_path 'DATA'
# one drive folder: --data_path "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/DATA/regression/DATA"
# LEONARDO: --data_path '/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA'


## Image channels: 1 size (240, 320)
python -m echocardiography.regression.train --data_path "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/DATA/regression/DATA" --save_dir /home/angelo/Desktop/treined \
                                            --target keypoints --model resnet50 \
                                            --input_channels 1 --size 240 320 \
                                            --epochs 50 --batch_size=32 --lr 0.001 --weight_decay 0.0 --threshold_wloss 0.0 