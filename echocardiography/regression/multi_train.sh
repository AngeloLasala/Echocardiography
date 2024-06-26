##  Image channels: 3 

# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.5
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.0001 --threshold_wloss 0.1
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.001 --threshold_wloss 0.1

## train the model for the direct regression
## direct regression - trial_24  GOOD
# python train.py --target heatmaps --epochs 50 --batch_size=32 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0

## train the model fro the heatmaps regression
## heatmaps regression - trial_25 GOOD
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.0001 --threshold_wloss 0.0 --model unet_up

## train the model for the heatmaps regression
## segmentation regression - trial_26 
# the threshold is set to 0.5 to avoid dealing with the class imbalance
# python train.py --target segmentation --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.0001 --threshold_wloss 0.0 --model unet_up

#################################################################################################################################################
## Image channels: 1

## train the model for the regression of heatmaps
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.001 --threshold_wloss 0.0 --model unet_up
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.0 --threshold_wloss 0.0 --model unet_up
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.00031 --weight_decay 0.0 --threshold_wloss 0.0 --model unet_up
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.0031 --weight_decay 0.0001 --threshold_wloss 0.0 --model unet_up

## train the model for the regression of coordinate
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --model resnet50
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.0031 --weight_decay 0.0 --threshold_wloss 0.0 --model resnet50
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.001 --weight_decay 0.0 --threshold_wloss 0.0 --model resnet50

# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --model swinv2
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.0031 --weight_decay 0.0 --threshold_wloss 0.0 --model swinv2
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.001 --weight_decay 0.0 --threshold_wloss 0.0 --model swinv2
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.0001 --weight_decay 0.0001 --threshold_wloss 0.0 --model swinv2
# python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.0001 --weight_decay 0.00001 --threshold_wloss 0.0 --model swinv2

## bigger nertwork
# python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --model resnet101
# python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --model resnet152
# python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --model swinv2_base

## Unet res - atnn
# python train.py --target heatmaps --epochs 50 --batch_size=4 --lr 0.0001 --weight_decay 0 --threshold_wloss 0.0 --model unet_res_skip_base
# python train.py --target heatmaps --epochs 50 --batch_size=4 --lr 0.001 --weight_decay 0 --threshold_wloss 0.0 --model unet_res_skip_base
# python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0 --threshold_wloss 0.0 --model unet_res_skip

#################################################################################################################################################
## Image channels: 1 size 221 x 295
python train.py --target keypoints --epochs 50 --batch_size=32 --lr 0.001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model resnet50
python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model resnet101
python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model resnet152

python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model swin_tiny
python train.py --target keypoints --epochs 50 --batch_size=16 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model swin_small
python train.py --target keypoints --epochs 50 --batch_size=8 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model swin_base

python train.py --target heatmaps --epochs 50 --batch_size=8 --lr 0.001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model unet_up
python train.py --target heatmaps --epochs 50 --batch_size=4 --lr 0.0001 --weight_decay 0.0 --threshold_wloss 0.0 --input_channels 1 --size 240 320 --model unet_res_skip







