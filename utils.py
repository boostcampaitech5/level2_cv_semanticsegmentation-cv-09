import random
import torch
import numpy as np
import os
import cv2 
import albumentations as A
from dataset.dataset import label2rgb
from inference import encode_mask_to_rle, decode_rle_to_mask
import torch.nn.functional as F

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
] 
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_gpu():
    GB = 1024.*1024.0*1024.0
    return round(torch.cuda.max_memory_allocated() / GB, 1)

def save_model(model, save_dir, file_name='fcn_resnet50_best_model.pt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path = os.path.join(save_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model, output_path)

def viz_img(path, model, thr):
    image = cv2.imread(path)
    tf = A.Resize(512, 512)
    image = image / 255.
    image = tf(image=image)['image']
    image = image.transpose(2, 0, 1)    # make channel first
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(dim=0)

    viz_outputs = model(image.to('cuda'))['out']
    viz_outputs = F.interpolate(viz_outputs, size=(2048, 2048), mode="bilinear")
    viz_outputs = torch.sigmoid(viz_outputs)
    viz_outputs = (viz_outputs > thr).detach().cpu().numpy() 

    viz_rles = []
    for output in viz_outputs:
        for segm in output:
            viz_rle = encode_mask_to_rle(segm)
            viz_rles.append(viz_rle)
        
    preds = []
    for rle in viz_rles[:len(CLASSES)]:
        pred = decode_rle_to_mask(rle, height=2048, width=2048)
        preds.append(pred)

    preds = np.stack(preds, 0)
    image = image.squeeze(dim=0).permute(1, 2, 0)
    return image, label2rgb(preds) 