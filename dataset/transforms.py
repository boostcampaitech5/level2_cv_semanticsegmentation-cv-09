import albumentations as A
import torch
import numpy as np


def get_train_transform(train=True,img_size=(512,512)):
    if train:
        transform = A.Compose([
            A.Resize(img_size[0],img_size[1]),
            A.Rotate(limit=10),
            A.HorizontalFlip(),

            ])
    else:
        transform = A.Resize(img_size[0],img_size[1])
    
    return transform


def get_test_transform(img_size=(512,512)):
    transform = A.Resize(img_size[0],img_size[1])
    
    return transform


def mixup_collate_fn(batch):
    indice = torch.randperm(len(batch))
    alpha = np.round(np.random.beta(0.2,0.2),2)
    img = []
    label = []
    for a, b in batch:
        img.append(a)
        label.append(b)
    img = torch.stack(img)
    label = torch.stack(label)

    shuffle_img = img[indice]
    shuffle_label = label[indice]
    
    img = alpha * img + (1-alpha) * shuffle_img
    label = alpha * label + (1-alpha) * shuffle_label

    return img,label
        