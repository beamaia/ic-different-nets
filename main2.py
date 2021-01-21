import torch
import torchvision 
from torchvision import models
import utils 
import data as dt
import train as tr

def main():
    model_name="resnet50"
    date="21-01-2021"
    version=2
    sets=3

    images_normal, images_carcinoma = dt.image_paths(First_Set=False, Both_Set=True, Hinton=True, home=False)
    x_normal = dt.process_images(images_normal)
    x_carcinoma = dt.process_images(images_carcinoma)
    images, labels = dt.create_images_labels(x_normal, x_carcinoma)
    (x_train, y_train), (x_test, y_test) = dt.create_train_test(images, labels, 0.25)
    train_loader, test_loader = dt.create_dataloaders(x_train, y_train, x_test, y_test)

    model = models.resnet50(pretrained=True)

    train_accuracies, train_losses, y_predict = tr.train(model, train_loader, num_epochs=100, lr=0.0001)
    test_accuracy, y_predict = tr.test(model, test_loader)
    print(train_accuracies[-1])
    print(test_accuracy)
    utils.save_y_true_predict(y_test, y_predict, model_name, date, version)
    utils.save_test_accuracy(test_accuracy, model_name, date, version, new=False, sets=sets) # change here

if __name__ == "__main__":
    main()