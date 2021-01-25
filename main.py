import torch
import torchvision
from torchvision import models
import torch.nn as nn

import numpy as np
from os import system

import data as dt
import train as tr
import utils

def get_info():
    sets = int(input("(1) First set, (2) Second Set, (3) Both \n"))
    epochs = int(input("Number of epochs:\n"))
    lr = float(input("Learning rate:\n"))

    first_set = False
    both_sets = False

    if sets == 1:
        first_set = True
    elif sets==3:
        both_sets = True

    dp = float(input("Dropout rate:\n"))
    model_name = input("Model name:\nShould be 'resnet50', 'mobilenetv2' or 'inceptionv3'\n")
    date = input("Date:\n")
    version = int(input("Version:\n"))

    hinton = (input("Hinton:\n"))
    if hinton == "True":
        hinton = True
    else:
        hinton = False

    bea = (input("Bea:\n"))
    if bea == "True":
        bea = True
    else:
        bea = False

    return first_set, both_sets, epochs, lr, sets, model_name, date, version, hinton, bea, dp

def main():

    first_set, both_sets, epochs, lr, sets, model_name, date, version, hinton, bea, dp = get_info()
    # images
    print("Processing images...")
    images_normal, images_carcinoma = dt.image_paths(First_Set=first_set, Both_Set= both_sets, Hinton=hinton, home=bea)
    x_normal = dt.process_images(images_normal)
    x_carcinoma = dt.process_images(images_carcinoma)
    images, labels = dt.create_images_labels(x_normal, x_carcinoma)

    print("Creating train, validation and test images...")

    # split
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dt.create_train_test(images, labels, 0.2)

    # dataloaders
    train_loader, val_loader, test_loader = dt.create_dataloaders(x_train, y_train, x_test, y_test, x_val, y_val)

    print("Configuring model...",end="\n\n")
    params = {
    'num_epochs': epochs,
    'lr': lr
    }
    
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dp), #change here
            nn.Linear(num_ftrs, 10)
        )
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        model.aux_logits=False
        # print(model)

    # train
    train_accuracies, train_losses, val_accuracies, val_losses, y_predict = tr.train(model, train_loader, val_loader, **params)

    print("Saving training data...", end="\n\n")
    
    # transforming into np.array
    array_train_accuracies = np.asarray(train_accuracies)
    array_train_losses = np.asarray(train_losses)
    array_val_accuracies = np.asarray(val_accuracies)
    array_val_losses = np.asarray(val_losses)

    # saving data
    # PATH = './saved_models/' + str(model_name) + "_" + str(date) + "_" + str(version)
    # torch.save(model.state_dict(), PATH)

    # test
    test_accuracy, y_predict = tr.test(model, test_loader)

    print("Saving test data...")

    utils.save_test_accuracy(test_accuracy, model_name, date, version, new=False, sets=sets) # change here
    utils.save_y_true_predict(y_test, y_predict, model_name, date, version)

    # Saving info
    utils.save_txt_accuracy_loss(array_train_accuracies, array_train_losses, date, model_name, version, training=True)
    utils.plot_accuracy_loss(epochs, model_name, losses=array_train_losses, accuracies=array_train_accuracies, date=date, version=version, training=True)

    # utils.save_txt_accuracy_loss(array_val_accuracies, array_val_losses, date, model_name, version, training=False)
    utils.plot_accuracy_loss(epochs, model_name, losses=array_val_losses, accuracies=array_val_accuracies, date=date, version=version, training=False)

if __name__ == "__main__":
    main()