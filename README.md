# Studying different neural networks

This repository contains a study on different popular convolutional neural networks (CNN) architectures pretrained on ImageNet and their use in histopathological images analysis. We will be using Pytorch's torchvision models subpacket. In this first analysis, we will study the models ResNet50, mobileNet-v3 and Inception-v3.

We'll be using the [histopathological imaging database for oral cancer analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6994517/) from Ayursundra Healthcare Pvt e Dr. B. Borooah Cancer Research Institute. 

At first, we will be using the First Set, composed of 528 images in 100x magnification, where 89 are images with no abnormalities and 439 images are of Oral Squamous Cell Carcinoma (OSCC). Afterwards, we will test the Second Set, composed of 696 images in 400x magnification, where 201 are images with no abnormalities and 495 images are of OSCC. And finally, we will test both sets together. Our data is divided into 70% for training, 15% for validation and 15% for testing.

## Neural Networks studied:

+ :heavy_check_mark: ResNet50
+ :x: MobileNet_v2
+ :x: Inception_v3

### Results
|Model name|Set|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|---|
|resnet50|First|98.96%|83.88%|12-01-2021|1|100|0.001|None|
|resnet50|First|98.52%|81.82%|13-01-2021|1|100|0.001|None|
|resnet50|First|99.68%|78.29%|14-01-2021|1|100|0.001|None|
|resnet50|Second|99.68%|76.85%|15-01-2021|2|100|0.0001|0.5|
