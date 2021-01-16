import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from glob import glob
import fnmatch
import cv2

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np

def image_paths(First_Set=True, Both_Set=False, Hinton=False, home=False):
    PATH_normal100x = 'First Set/100x Normal Oral Cavity Histopathological Images/*'
    PATH_carcinoma100x = 'First Set/100x OSCC Histopathological Images/*'
    PATH_normal400x = 'Second Set/400x Normal Oral Cavity Histopathological Images/*'
    PATH_carcinoma400x = 'Second Set/400x OSCC Histopathological Images/*'
    
    if home:
        PATH_ini = ".././data/oralCancer-Borooah/"
    elif Hinton:
        PATH_ini = "/home/hinton/Desktop/Humberto/"
    else:
        PATH_ini = "./data/oralCancer-Borooah/"

    if not Both_Set:
        if First_Set:
            PATH_normal = PATH_ini + PATH_normal100x
            PATH_carcinoma = PATH_ini + PATH_carcinoma100x
        else:
            PATH_normal = PATH_ini + PATH_normal400x
            PATH_carcinoma = PATH_ini + PATH_carcinoma400x
        
        images_normal = glob(PATH_normal)
        images_carcinoma = glob(PATH_carcinoma)

    else:
        PATH_normal100 = PATH_ini + PATH_normal100x
        PATH_carcinoma100 = PATH_ini + PATH_carcinoma100x
        PATH_normal400 = PATH_ini + PATH_normal400x
        PATH_carcinoma400 = PATH_ini + PATH_carcinoma400x
        images_normal100 = glob(PATH_normal100)
        images_carcinoma100 = glob(PATH_carcinoma100)
        images_normal400 = glob(PATH_normal400)
        images_carcinoma400 = glob(PATH_carcinoma400)

        images_normal = images_normal100 + images_normal400
        images_carcinoma = images_carcinoma100 + images_carcinoma400

    print(f"Found {len(images_normal)} images of class: normal and {len(images_carcinoma)} images of class: carcinoma", end="\n\n")
    return images_normal, images_carcinoma

def process_images(images):
    height, width = 250, 250
    
    x = []
        
    for _, img in enumerate(images):
        full_size_image = cv2.imread(img)
        image = (cv2.resize(full_size_image, (width, height), interpolation = cv2.INTER_CUBIC))
        
        x.append(image)

    return np.array(x)

def create_images_labels(x_normal, x_carcinoma):
    # Creating patches
    pe = PatchExtractor(patch_size = (30, 30), max_patches = 30)
    patches_normal = pe.transform(x_normal)
    patches_carcinoma = pe.transform(x_carcinoma)
    images = np.concatenate((patches_normal, patches_carcinoma), axis = 0)

    labels = []

    labels_nr = [0 for i,_ in enumerate(patches_normal)]
    labels_ca = [1 for i,_ in enumerate(patches_carcinoma)]
    labels = labels_nr + labels_ca

    labels = np.array(labels)

    images = torch.from_numpy(images)
    labels = torch.Tensor(labels) 

    images = images.type(torch.FloatTensor) 
    labels = labels.type(torch.LongTensor) 

    return images, labels

def create_train_test(images, labels, test_size):
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size =0.5, random_state=42)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# DataLoader

class DatasetOral(torch.utils.data.Dataset):
    def __init__(self, images, labels, transformation):
        self.labels = labels
        self.images = images
        self.transformation = transformation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

def create_dataloaders(x_train, y_train, x_test, y_test, x_val, y_val, batch_size=64, shuffle=True, num_workers=2):
    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': num_workers}

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    x_train = x_train.reshape(-1, 3, 30, 30)
    x_val = x_val.reshape(-1, 3, 30, 30)
    x_test = x_test.reshape(-1, 3, 30, 30)

    train_set = DatasetOral(x_train, y_train, transform)
    train_loader = DataLoader(train_set, **params)

    val_set = DatasetOral(x_val, y_val, transform)
    val_loader = DataLoader(val_set, **params)

    test_set = DatasetOral(x_test, y_test, transform)
    test_loader = DataLoader(test_set, **params)

    return train_loader, val_loader, test_loader

