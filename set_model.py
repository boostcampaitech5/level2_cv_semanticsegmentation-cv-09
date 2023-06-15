import torch
import torch.nn as nn
import os
import argparse
import cv2 
import bentoml
import sys



class wrapper_model(nn.Module):
    def __init__(self, model, resize):
        super().__init__()
        self.model = model 
        self.resize = resize
    def forward(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, self.resize)
        image = image / 255.
        image = image.transpose(2, 0, 1)    # make channel first
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(dim=0)
        
        return self.model(image)
    
def get_argparser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './checkpoint'))
    parser.add_argument('--weight', type=str, default='best.pt')
    parser.add_argument("--resize", nargs="+", type=int, default=[512, 512], help='resize size for image when training')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_argparser()
    sys.path.insert(0, os.path.join(args.model_dir, args.weight))
    
    model = torch.load(os.path.join(args.model_dir, args.weight))
    bentoml.pytorch.save_model(
    "pytorch_Unet",
    wrapper_model(model, args.resize),
    signatures={"__call__": {"batchable": True, "batch_dim": 0}}
)