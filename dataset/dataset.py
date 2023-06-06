from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
from sklearn.model_selection import GroupKFold
import json

def split_dataset(img_dir="/opt/ml/train/DCM",label_dir="/opt/ml/train/outputs_json"):
    
    # img_dir 아래 폴더를 순회하면서 .png 파일들을 찾습니다
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=img_dir)
        for root, _dirs, files in os.walk(img_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    
    # label_dir 아래에 있는 모든 폴더를 재귀적으로 순회하면서 json을 찾습니다.
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_dir)
        for root, _dirs, files in os.walk(label_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    # 모든 .png파일에 대해 .json pair가 존재하는지 체크합니다
    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
    
    # 이름 순으로 정렬해서 짝지 맞도록 합니다
    pngs = sorted(pngs)
    jsons = sorted(jsons)   
    
    _filenames = np.array(pngs)
    _labelnames = np.array(jsons)
    
    # split train-valid
    # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
    # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
    # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
    groups = [os.path.dirname(fname) for fname in _filenames]
    
    # dummy label   
    ys = [0 for _ in _filenames]
    
    # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
    # 5으로 설정하여 KFold를 수행합니다.
    gkf = GroupKFold(n_splits=5)
    

    for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        # x를 train, y를 val
        val_filenames = list(_filenames[y])
        val_labelnames = list(_labelnames[y])
        
        train_filenames = list(_filenames[x])
        train_labelnames = list(_labelnames[x])
            
        # skip i > 0
        break
    
    return train_filenames, train_labelnames, val_filenames, val_labelnames


class XRayDataset(Dataset):
    
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
    
    PALETE = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
        ]
    
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    
    def __init__(self,
                 filenames,
                 labelnames,
                 data_dir="/opt/ml/train/DCM",
                 label_dir="/opt/ml/train/outputs_json",
                 is_train=True,
                 transforms=None):
        
        self.image_root = data_dir
        self.label_dir = label_dir
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_dir, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(self.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label


class XRayInferenceDataset(Dataset):
    
    CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    
    PALETE = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
        ]
    
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    
    def __init__(self, 
                 data_dir="/opt/ml/test/DCM",
                 transforms=None):
        
        self.data_dir = data_dir
        
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.data_dir)
            for root, _dirs, files in os.walk(self.data_dir)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.data_dir, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name
    
    
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = XRayDataset.PALETE[i]
        
    return image