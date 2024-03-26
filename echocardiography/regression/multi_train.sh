# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.5
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.0001 --threshold_wloss 0.1
python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.001 --threshold_wloss 0.1

## train the model for the direct regression
## direct regression - trial_24  GOOD
python train.py --target heatmaps --epochs 50 --batch_size=32 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0

## train the model fro the heatmaps regression
## heatmaps regression - trial_25 GOOD
python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.0001 --threshold_wloss 0.0 --model unet_up

## train the model for the heatmaps regression
## segmentation regression - trial_26 
# the threshold is set to 0.5 to avoid dealing with the class imbalance
python train.py --target segmentation --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.0001 --threshold_wloss 0.0 --model unet_up
