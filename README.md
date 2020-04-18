# Road Scene Segmentation
These two folders represent two networks that are used in Semantic Segmentation in road sences.

1- [FCHarDNet](https://github.com/PingoLH/FCHarDNet)
+ 2- [ESPNetV2](https://github.com/sacmehta/EdgeNets)

## Weights

For FCHarDNet: https://ping-chao.com/hardnet/hardnet70_cityscapes_model.pkl
For ESPNetV2: Present inside https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/ESPNetv2/model/segmentation/model_zoo/espnetv2

#### Note:
> These two networks use CityScapes dataset to segment objects - Don't use other datasets' pretrained models - in the frame but were tested on KITTI dataset.

## Weight Placement

For FCHarDNet: Place "hardnet70_cityspaces_model.pkl" inside the FCHarDNet folder so that it is in the same directory as ["hardnet.py"](https://github.com/omarsayed7/Road-Scene-Segmentation/blob/master/FCHarDNet/hardnet.py)
For ESPNetV2: Already present.

## Usage

- FCHarDNet: While opening [this](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/FCHarDNet) path, run:
'''
python3 hardnet.py
'''
- ESPNetV2: While opening [this](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/ESPNetv2) path, run:
'''
python3 espnet_v2.py
'''
## Results
