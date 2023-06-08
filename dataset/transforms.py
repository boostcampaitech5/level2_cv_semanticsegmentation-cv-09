import albumentations as A

def get_train_transform(train=True):
    if train:
        transform = A.Compose([
            A.Resize(512,512),
            A.Rotate((-10, 10), p=0.5),
            A.HorizontalFlip(p=0.5)
            ])
    else:
        transform = A.Resize(512,512)
    
    return transform


def get_test_transform():
    transform = A.Resize(512,512)
    
    return transform