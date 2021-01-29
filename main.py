import torch
from torchvision import models
import torch.nn as nn

import numpy as np

from os import system
import sys
from datetime import date

import data as dt
import train as tr
import utils

def get_arg(args):
    if len(args) != 9:
        sys.exit(1)

    set_numb = int(sys.argv[1])
    epochs = int(sys.argv[2])
    lr = float(sys.argv[3])
    model_name = sys.argv[4] 
    dp = float(sys.argv[5])
    version = int(sys.argv[6])
    server = sys.argv[7] 
    hw = int(sys.argv[8])
    
    today = date.today()
    date_today = today.strftime("%d-%m-%y")

    return set_numb, epochs, lr, model_name, dp, version, server, date_today, hw

def main(date_today, set_numb=1, epochs=200, lr=0.0001, model_name="resnet50", dp=0.5, version=1, server="hinton", hw=224):
    # set_numb, epochs, lr, dp, model_name,  date_today, version, server = get_info()
    
    # images    
    images_normal, images_carcinoma = dt.image_paths(set_numb, server)
    x_normal = dt.process_images(images_normal,hw,hw)
    x_carcinoma = dt.process_images(images_carcinoma,hw,hw)
    images, labels = dt.create_images_labels(x_normal, x_carcinoma)

    # split
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dt.create_train_test(images, labels, 0.2)

    # dataloaders
    train_loader, val_loader, test_loader = dt.create_dataloaders(x_train, y_train, x_test, y_test, x_val, y_val, 32)

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
        model = models.inception_v3(pretrained=True, aux_logits = False)
        # print(model)

    else:
        print("Model not identified")
        sys.exit(1)


    # train
    train_accuracies, train_losses, val_accuracies, val_losses, y_predict = tr.train(model, model_name, train_loader, val_loader, **params)

    print("Saving training data...", end="\n\n")
    
    # transforming into np.array
    array_train_accuracies = np.asarray(train_accuracies)
    array_train_losses = np.asarray(train_losses)
    array_val_accuracies = np.asarray(val_accuracies)
    array_val_losses = np.asarray(val_losses)

    # save_model_dict (model, model_name,  date_today, version)

    # test
    test_accuracy, y_predict = tr.test(model, test_loader)

    print("Saving test data...")

    utils.save_test_accuracy(test_accuracy, model_name, date_today, version, new=False, sets=set_numb) # change here
    utils.save_y_true_predict(y_test, y_predict, model_name, date_today, version)

    # Saving info
    utils.save_txt_accuracy_loss(array_train_accuracies, array_train_losses, date_today, model_name, version, training=True)
    
    print("Plotting images...")
    utils.plot_accuracy_loss(epochs, model_name, losses=array_train_losses, accuracies=array_train_accuracies, date=date_today, version=version, training=True)

    # utils.save_txt_accuracy_loss(array_val_accuracies, array_val_losses, date, model_name, version, training=False)
    utils.plot_accuracy_loss(epochs, model_name, losses=array_val_losses, accuracies=array_val_accuracies, date=date_today, version=version, training=False)

if __name__ == "__main__":
    print("Starting Program!")
    arg = sys.argv
    set_numb, epochs, lr, model_name, dp, version, server, date_today, hw = get_arg(arg)
    main(date_today, set_numb=set_numb, epochs=epochs, lr=lr, model_name=model_name.lower(), dp=dp, version=version, server=server, hw=hw)
    print("Program ended!")