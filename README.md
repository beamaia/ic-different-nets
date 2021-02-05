# Studying different neural networks

This repository contains a study on different popular convolutional neural networks (CNN) architectures pretrained on ImageNet and their use in histopathological images analysis. We will be using Pytorch's torchvision models subpacket. In this first analysis, we will study the models ResNet50, mobileNet-v3 and Inception-v3.

We'll be using the [histopathological imaging database for oral cancer analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6994517/) from Ayursundra Healthcare Pvt e Dr. B. Borooah Cancer Research Institute. 

At first, we will be using the First Set, composed of 528 images in 100x magnification, where 89 are images with no abnormalities and 439 images are of Oral Squamous Cell Carcinoma (OSCC). Afterwards, we will test the Second Set, composed of 696 images in 400x magnification, where 201 are images with no abnormalities and 495 images are of OSCC. And finally, we will test both sets together.

## Neural Networks studied:

+ :o: ResNet50
+ :o: MobileNet_v2
+ :o: Inception_v3

### Results

Results are ordered by dropout rate, model, and number of epochs, and divided by sets used. 
January tests: unweighted dataloaders 
February tests: with weighted samplers

#### First Set

|Model|Train Accuracy|Test Accuracy|Test Balanced Accuracy|Precision|Recall|F1|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|---|---|---|---|
|resnet50|99.76%|90.57%|50%|0.7736|1.0000|0.8723|05-02-2021|1|200|0.0001|0.5|
