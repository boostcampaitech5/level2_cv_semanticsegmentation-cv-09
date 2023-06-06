import albumentations as A

def get_train_transform(val=False):
    if val:
        transform = A.Compose([
            A.Resize(512,512),
            ])
    else:
        transform = A.Resize(512,512)
    
    return transform


def get_test_transform():
    transform = A.Resize(512,512)
    
    return transform