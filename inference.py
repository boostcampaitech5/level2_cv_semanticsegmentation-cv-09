import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import XRayInferenceDataset
from dataset.transforms import get_test_transform
import os
import pandas as pd
import argparse
from importlib import import_module
import torch.nn.functional as F

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(XRayInferenceDataset.CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            # outputs = (outputs > 0.1).detach().cpu().numpy()
            for i in range(29):
                if i==26 or i == 20:
                    outputs[:,i] = (outputs[:,i]> 0.9)
                else:
                    outputs[:,i] = (outputs[:,i]> 0.5)
            outputs = outputs.detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{XRayInferenceDataset.IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


def get_argparser():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for validing (default: 1000)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/test/DCM'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './checkpoint'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--weight', type=str, default='best.pt')
    parser.add_argument("--resize", nargs="+", type=int, default=[512, 512], help='resize size for image when training')

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = get_argparser()
    
    print(args.resize)
    transform = get_test_transform(args.resize)
    
    test_dataset = XRayInferenceDataset(
        data_dir=args.data_dir,
        transforms=transform
        )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    model = torch.load(os.path.join(args.model_dir, args.weight))

    
    rles, filename_and_class = test(model, test_loader)
    
    # To CSV
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
      
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    df.to_csv(os.path.join(args.output_dir,"output.csv"), index=False)