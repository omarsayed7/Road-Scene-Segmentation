# Road Scene Segmentation
These two folders represent two networks that are used in Semantic Segmentation in road sences.

+ [FCHarDNet](https://github.com/PingoLH/FCHarDNet)
+ [ESPNetV2](https://github.com/sacmehta/EdgeNets)

## Pretrained Models

+ [FCHarDNet](https://ping-chao.com/hardnet/hardnet70_cityscapes_model.pkl)
+ [ESPNetV2](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/ESPNetv2/model/segmentation/model_zoo/espnetv2)

## Usage
+ For FCHarDNet: Create a folder named "pretrained" in the same directory as ["hardnet.py"](https://github.com/omarsayed7/Road-Scene-Segmentation/blob/master/FCHarDNet/hardnet.py) and place "hardnet70_cityspaces_model.pkl" inside 
- FCHarDNet: While opening [this](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/FCHarDNet) path, run:
```
python3 hardnet.the s
```
- ESPNetV2: While opening [this](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/ESPNetv2) path, run:
```
python3 espnet_v2.py
```
## Results
+ [Sample image](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/ESPNetv2/sample_images)
+ [FCHarDNet](https://drive.google.com/open?id=1OYWKakLbSPzsFdDBKI7RewiCpJ1hOw-j)
+ [ESPNetV2](https://github.com/omarsayed7/Road-Scene-Segmentation/tree/master/ESPNetv2/segmentation_results)

Note: All the codes are forked from [FCHarDNet](https://www.google.com/search?channel=fs&client=ubuntu&q=hardnet+github) and [ESPNetv2](https://github.com/sacmehta/ESPNetv2)
