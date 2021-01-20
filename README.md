# Studying different neural networks

This repository contains a study on different popular convolutional neural networks (CNN) architectures pretrained on ImageNet and their use in histopathological images analysis. We will be using Pytorch's torchvision models subpacket. In this first analysis, we will study the models ResNet50, mobileNet-v3 and Inception-v3.

We'll be using the [histopathological imaging database for oral cancer analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6994517/) from Ayursundra Healthcare Pvt e Dr. B. Borooah Cancer Research Institute. 

At first, we will be using the First Set, composed of 528 images in 100x magnification, where 89 are images with no abnormalities and 439 images are of Oral Squamous Cell Carcinoma (OSCC). Afterwards, we will test the Second Set, composed of 696 images in 400x magnification, where 201 are images with no abnormalities and 495 images are of OSCC. And finally, we will test both sets together.

## Neural Networks studied:

+ :o: ResNet50
+ :x: MobileNet_v2
+ :x: Inception_v3

### Results

Results are ordered by dropout rate and number of epochs, and divided by sets used. 

For data divided into 70% for training, 15% for validation and 15% for testing:

#### First Set

|Model|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|
|resnet50|98.96%|83.88%|12-01-2021|1|100|0.001|None|
|resnet50|98.52%|81.82%|13-01-2021|1|100|0.001|None|
|resnet50|99.68%|78.29%|14-01-2021|1|100|0.001|None|
|resnet50|99.68%|82.91%|15-01-2021|3|100|0.0001|0.2|
|resnet50|99.37%|83.33%|15-01-2021|4|100|0.0001|0.2|
|resnet50|99.76%|85.14%|16-01-2021|3|100|0.0001|0.5|
|resnet50|99.56%|85.86%|16-01-2021|4|100|0.0001|0.5|

#### Second Set

|Model|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|
|resnet50|99.56%|77.23%|15-01-2021|1|100|0.0001|None|
|resnet50|99.63%|76.12%|16-01-2021|1|100|0.0001|0.2|
|resnet50|99.73%|76.31%|16-01-2021|5|100|0.0001|0.2|
|resnet50|99.47%|76.85%|15-01-2021|2|100|0.0001|0.5|
|resnet50|99.66%|65.33%|16-01-2021|7|100|0.0001|0.5|
|resnet50|99.51%|77.46%|19-01-2021|1|100|0.0001|0.5|

#### Both Sets

|Model|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|
|resnet50|99.42%|79.58%|19-01-2021|3|100|0.0001|None|
|resnet50|99.63%|75.60%|16-01-2021|2|100|0.0001|0.2|
|resnet50|99.43%|77.74%|16-01-2021|6|100|0.0001|0.2|
|resnet50|99.44%|74.73%|18-01-2021|1|100|0.0001|0.5|
|resnet50|99.56%|77.07%|19-01-2021|2|100|0.0001|0.5|

For data divided into 80% for training, 10% for validation and 10% for testing:

#### First Set

|Model|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|
|resnet50|99.64%|87.12%|19-01-2021|5|100|0.001|None|
|resnet50|99.75%|85.54%|19-01-2021|6|100|0.001|None|

#### Second Set

|Model|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|
|resnet50|99.55%|79.84%|20-01-2021|1|100|0.0001|None|
|resnet50|%|%|20-01-2021|2|100|0.0001|0.2|

<!-- #### Both Sets

|Model|Train Accuracy|Test Accuracy|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|
|resnet50|99.42%|79.58%|19-01-2021|3|100|0.0001|None| -->
<!-- |resnet50|99.63%|75.60%|16-01-2021|2|100|0.0001|0.2| -->

Key:
+ Model: name of the model used;
+ Train Accuracy: accuracy of the last epoch while training;
+ Test Accuracy: accuracy while testing;
+ Date: date the model was trained;
+ Version: training attempt number for that model on that day;
+ Epoch: number of epochs used;
+ LR: learning rate used (fixed);
+ Dropout: value if dropout layer was added, or "None" if wasn't used.