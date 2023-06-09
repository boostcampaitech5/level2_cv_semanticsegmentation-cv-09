import albumentations as A

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


def get_test_transform():
    transform = A.Resize(512,512)
    
    return transform