
# importing libraries
import torch
import torchvision
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import utils

from sklearn import metrics

def train (model, model_name, train_loader, val_loader, weights, num_epochs, lr):
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # weights = weights.to(device)
    print(f"Device: {device}",end="\n\n")

    criterion = nn.CrossEntropyLoss() #Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_accuracies = []
    train_losses = []
    y_predict = []
    val_accuracies = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        # Model in training mode
        model.train()
    
        running_loss = 0
        total_train = 0
        accuracies_train = 0
        y_predict_loader = []
        
        for _, data in enumerate(train_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            #Forward
            outputs = model(images)

            loss = criterion(outputs, labels).to(device) 
            loss.backward()

            #Optimize
            optimizer.step()

            running_loss += loss.item()

            #Train accuracy
            _, predicted = torch.max(outputs, 1)
            y_predict_loader.append(predicted)
            total_train += labels.size(0)
            accuracies_train += (predicted == labels).sum().item()

        accuracy = accuracies_train / total_train * 100
        train_losses.append(np.mean(running_loss)) 
        train_accuracies.append(accuracy)
        y_predict.append(y_predict_loader)

        print(f"Accuracy: {accuracy:.2f} %")
        print(f"Loss: {np.mean(running_loss):.2f}")

        # Validation 
        val_loss = 0
        total_val = 0
       
        with torch.no_grad():
            accuracies_val = 0  
            running_val_loss = 0
            for _, data in enumerate(val_loader):
                # images, labels = data[0], data[1]
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                val_loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                total_val += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                accuracies_val += (predicted == labels).sum().item()
                
        
        accuracy_val = accuracies_val / total_val * 100

        print(f"Val Accuracy: {accuracy_val:.2f} %")
        print(f"Val Loss: {np.mean(running_val_loss):.2f}", end="\n\n")
        
        # Saving validation and 
        val_losses.append(np.mean(running_val_loss))
        val_accuracies.append(accuracy_val)

    print("Finished training")

    return train_accuracies, train_losses, val_accuracies, val_losses, y_predict

def test(model, test_loader):
    print("Starting test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    y_predict = []
    y_true = []
    accuracy = 0
    with torch.no_grad():
        total = 0
        for _, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            y_true.append(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_predict.append(predicted)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy_test = accuracy / total * 100
    print(f"Accuracy: {accuracy_test}")
    print("Finished test....")
    return accuracy_test, y_predict, y_true