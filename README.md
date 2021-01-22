# Studying different neural networks

This repository contains a study on different popular convolutional neural networks (CNN) architectures pretrained on ImageNet and their use in histopathological images analysis. We will be using Pytorch's torchvision models subpacket. In this first analysis, we will study the models ResNet50, mobileNet-v3 and Inception-v3.

We'll be using the [histopathological imaging database for oral cancer analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6994517/) from Ayursundra Healthcare Pvt e Dr. B. Borooah Cancer Research Institute. 

At first, we will be using the First Set, composed of 528 images in 100x magnification, where 89 are images with no abnormalities and 439 images are of Oral Squamous Cell Carcinoma (OSCC). Afterwards, we will test the Second Set, composed of 696 images in 400x magnification, where 201 are images with no abnormalities and 495 images are of OSCC. And finally, we will test both sets together.

## Neural Networks studied:

+ :o: ResNet50
+ :o: MobileNet_v2
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

|Model|Train Accuracy|Test Accuracy|Precision|Recall|F1|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|---|---|---|
|resnet50|99.64%|87.12%|0.8456|0.9149|0.8789|19-01-2021|5|100|0.001|None|
|resnet50|99.75%|85.54%|0.8490|0.8635|0.8562|19-01-2021|6|100|0.001|None|
|mobilenet_v2|99.46%|85.54%|0.8504|0.8561|0.8532|22-01-2021|1|200|0.001|None|
|mobilenet_v2|99.43%|85.80%|0.8472|0.9060|0.8757|22-01-2021|2|200|0.001|None|
|resnet50|99.63%%|86.17%|0.8452|0.8837|0.8640|21-01-2021|3|200|0.001|0.5|

#### Second Set

|Model|Train Accuracy|Test Accuracy|Precision|Recall|F1|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|---|---|---|
|resnet50|99.55%|79.84%|0.7139|0.7612|0.7368|20-01-2021|1|100|0.0001|None|
|resnet50|99.25%|74.76%|0.7154|0.7043|0.7098|20-01-2021|2|100|0.0001|None|
|mobilenet_v2|99.26%|76.58%|0.7183|0.7525|0.7355|22-01-2021|3|200|0.001|None|

#### Both Sets

|Model|Train Accuracy|Test Accuracy|Precision|Recall|F1|Date|Version|Epoch|LR|Dropout|
|---|---|---|---|---|---|---|---|---|---|---|
|resnet50|99.49%|79.14%|0.7669|0.8328|0.7985|20-01-2021|3|100|0.0001|None|
|resnet50|99.52%|78.81%|0.7588|0.7931|0.7756|20-01-2021|4|100|0.0001|None|

Key:
+ Model: name of the model used;
+ Train Accuracy: accuracy of the last epoch while training;
+ Test Accuracy: accuracy while testing;
+ Date: date the model was trained;
+ Version: training attempt number for that model on that day;
+ Epoch: number of epochs used;
+ LR: learning rate used (fixed);
+ Dropout: value if dropout layer was added, or "None" if wasn't used.